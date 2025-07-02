# core/generation_utils.py
# Contains helper functions refactored from generation_core.py
import logging
import os
import json
import io
import zipfile
import torch
from PIL import Image
import numpy as np
from typing import Optional, Tuple, Set, Dict

from ui.shared_state import shared_state_instance
from diffusers_helper.memory import load_model_as_complete, unload_complete_models, gpu
from diffusers_helper.hunyuan import vae_decode
from diffusers_helper.utils import save_bcthw_as_mp4, generate_timestamp
from diffusers_helper.gradio.progress_bar import make_progress_bar_html

from . import generation_utils
logger = logging.getLogger(__name__)

def generate_roll_off_schedule(
    total_steps: int,
    peak_cfg: float,
    final_cfg: float,
    roll_off_start_percent: float,
    roll_off_factor: float
) -> list[float]:
    """
    Generates a CFG schedule that holds a peak value and then rolls off.

    Args:
        total_steps (int): The total number of inference steps.
        peak_cfg (float): The CFG value to hold before the roll-off.
        final_cfg (float): The CFG value to ramp down to at the final step.
        roll_off_start_percent (float): The point (0.0 to 1.0) to start the roll-off.
        roll_off_factor (float): The exponential factor for the curve (1.0 is linear).

    Returns:
        list[float]: A list of CFG values, one for each step.
    """
    # Calculate the step at which the ramp-down begins
    ramp_down_start_step = int(round(total_steps * roll_off_start_percent))
    
    # The number of steps to hold the peak CFG
    sustain_steps = ramp_down_start_step
    
    # The number of steps for the ramp-down phase
    ramp_steps = total_steps - sustain_steps
    
    if ramp_steps <= 0:
        # If the start point is at or after 100%, just hold the peak CFG
        return np.full(total_steps, peak_cfg).tolist()

    # Phase 1: Hold the peak CFG
    sustain_phase = np.full(sustain_steps, peak_cfg)

    # Phase 2: Generate the roll-off curve
    # Create a normalized time vector from 0 to 1 for the ramp
    t = np.linspace(0, 1, ramp_steps)
    # Apply the roll-off factor to shape the curve
    t_curved = t ** roll_off_factor
    
    # Interpolate from peak to final CFG along the curved timeline
    roll_off_phase = peak_cfg + (final_cfg - peak_cfg) * t_curved

    # Combine the phases and return
    full_schedule = np.concatenate([sustain_phase, roll_off_phase])
    
    return full_schedule.tolist()


def initialize_job(
    total_second_length: float,
    fps: int,
    latent_window_size: int,
    task_id: str,
    output_queue_ref,
) -> Tuple[int, str]:
    """
    Calculates job parameters, creates a job ID, and sends an initial progress update.
    This is the first "secret sauce" block.
    """
    # A "segment" or "section" is one generation loop, which produces (LWS * 4 - 3) frames.
    total_frames = int(total_second_length * fps)
    frames_per_segment = latent_window_size * 4 - 3
    total_latent_sections = (
        int(max(round(total_frames / frames_per_segment), 1))
        if frames_per_segment > 0
        else 1
    )

    job_id = f"{generate_timestamp()}_task{task_id}"
    output_queue_ref.push(
        (
            "progress",
            (
                task_id,
                None,
                f"Total Segments: {total_latent_sections}",
                make_progress_bar_html(0, "Starting ..."),
            ),
        )
    )
    return total_latent_sections, job_id


def handle_segment_saving(
    # Loop state
    latent_padding_iteration: int,
    is_last_section: bool,
    current_loop_segment_number: int,
    total_latent_sections: int,
    current_video_frame_count: int,
    history_pixels,
    # Job/Task info
    task_id: str,
    job_id: str,
    output_queue_ref,
    outputs_folder: str,
    # User settings
    preview_frequency: int,
    parsed_segments_to_decode_set: Set[int],
    fps: int,
    mp4_crf: int,
    force_standard_fps: bool = False
) -> Optional[str]:
    """
    Handles the logic for saving an MP4 for the current segment, either automatically
    or by user request. This is the second "secret sauce" block.
    Returns the path to the saved file, or None if no file was saved.
    """
    # First, check if a preview has been manually requested by the user.
    is_preview_request = shared_state_instance.preview_request_flag.is_set()
    if is_preview_request:
        # IMPORTANT: Clear the flag immediately after checking. This is the core of the
        # "graceful preview" feature, ensuring the worker continues generation.
        shared_state_instance.preview_request_flag.clear()

    # Next, determine if we should save based on any of the automatic criteria.
    should_save_automatically = (
        latent_padding_iteration == 0  # Always save the very first segment
        or is_last_section  # Always save the final completed video
        or (
            parsed_segments_to_decode_set
            and current_loop_segment_number in parsed_segments_to_decode_set
        )  # Save if the user specified this segment number
        or (
            preview_frequency > 0 and (current_loop_segment_number % preview_frequency == 0)
        )  # Save based on the periodic preview_frequency setting
    )

    if is_preview_request or should_save_automatically:
        save_hint = "Saving Preview..." if is_preview_request else "Saving Segment..."
        if is_last_section:
            save_hint = "Saving Final Video..."

        output_queue_ref.push(("progress", (task_id, None, f"Segment {current_loop_segment_number}/{total_latent_sections}: {save_hint}", make_progress_bar_html(100, save_hint))))

        segment_mp4_filename = os.path.join(outputs_folder, f"{job_id}_segment_{current_loop_segment_number}_frames_{current_video_frame_count}.mp4")
        save_bcthw_as_mp4(history_pixels, segment_mp4_filename, fps=fps, crf=mp4_crf)
        
        logger.info(f"Task {task_id}: SAVED MP4 for segment {current_loop_segment_number} to {segment_mp4_filename}. Total video frames: {current_video_frame_count}")
        output_queue_ref.push(("file", (task_id, segment_mp4_filename, f"Segment {current_loop_segment_number} MP4 saved ({current_video_frame_count} frames)")))
        return segment_mp4_filename
    else:
        logger.info(f"Task {task_id}: SKIPPED MP4 save for intermediate segment {current_loop_segment_number}.")
        return None

def save_resume_state(
    outputs_folder: str,
    job_id: str,
    current_loop_segment_number: int,
    history_latents: torch.Tensor,
    input_image_np: np.ndarray,
    creative_params: Dict,
    task_id: str,
    retention_count: int,
) -> Optional[str]:
    """
    Saves the current generation state to a .goan_resume zip archive.
    This is the third "secret sauce" block.
    """
    resume_dir = os.path.join(outputs_folder, "resume_states")
    os.makedirs(resume_dir, exist_ok=True)

    if retention_count == 0:
        return None # Do not save if retention is explicitly zero.

    resume_filename_base = f"{job_id}_resume_seg_{current_loop_segment_number}"
    resume_zip_path = os.path.join(resume_dir, f"{resume_filename_base}.goan_resume")

    logger.info(f"Task {task_id}: Saving resume state to {resume_zip_path}...")

    try:
        with zipfile.ZipFile(resume_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # 1. Save creative parameters to params.json
            # The passed dict contains all necessary creative & environment params.
            zf.writestr('params.json', json.dumps(creative_params, indent=4))

            # 2. Save source image to source_image.png
            img = Image.fromarray(input_image_np)
            with io.BytesIO() as buf:
                img.save(buf, format='PNG')
                zf.writestr('source_image.png', buf.getvalue())

            # 3. Save latent history to latent_history.pt
            with io.BytesIO() as buf:
                torch.save(history_latents.cpu(), buf)
                zf.writestr('latent_history.pt', buf.getvalue())

            # 4. Save job metadata to job_info.json
            job_info = {
                'job_id': job_id,
                'completed_segments': current_loop_segment_number,
            }
            zf.writestr('job_info.json', json.dumps(job_info, indent=4))

        logger.info(f"Task {task_id}: Successfully saved resume state to {resume_zip_path}")

        # --- Handle rolling file retention ---
        if retention_count > 0:
            try:
                # Find all resume files for this job_id
                all_resume_files = [f for f in os.listdir(resume_dir) if f.startswith(job_id) and f.endswith('.goan_resume')]
                if len(all_resume_files) > retention_count:
                    # Sort files to find the oldest ones. Sorting by name works due to the _seg_N suffix.
                    all_resume_files.sort(key=lambda name: int(name.split('_seg_')[1].split('.')[0]))
                    files_to_delete = all_resume_files[:-retention_count]
                    for f_del in files_to_delete:
                        os.remove(os.path.join(resume_dir, f_del))
                        logger.info(f"Task {task_id}: Deleted old resume file: {f_del}")
            except Exception as e_clean:
                logger.error(f"Task {task_id}: Error during resume file cleanup: {e_clean}", exc_info=True)
        return resume_zip_path
    except Exception as e:
        logger.error(f"Task {task_id}: Failed to save resume state: {e}", exc_info=True)
        return None


def _save_final_preview(history_latents, vae, job_id, task_id, outputs_folder, crf, fps, output_queue_ref, high_vram):
    """
    Helper function to decode and save the final video preview during a graceful abort.
    This logic is refactored to be callable from the end of the worker.
    """
    if history_latents is None:
        logger.warning(f"Task {task_id}: No latents generated, cannot save final preview.")
        return None

    # Check for a hard abort (level 2) before starting the expensive decode.
    # This allows a double-click abort to interrupt the graceful save.
    if shared_state_instance.abort_state['level'] >= 2:
        logger.warning(f"Task {task_id}: Hard abort detected before final VAE decode.")
        raise InterruptedError("Hard abort during final save.")

    logger.info(f"Task {task_id}: Decoding final latents for graceful abort preview...")
    output_queue_ref.push(('progress', (task_id, None, "Decoding final latents for preview...", make_progress_bar_html(100, "Decoding..."))))

    if not high_vram:
        load_model_as_complete(vae, target_device=gpu)

    # This is a blocking, expensive operation, necessary only to produce mp4.
    pixels = vae_decode(history_latents, vae).cpu()

    if not high_vram:
        unload_complete_models(vae)

    # Add a second check for a hard abort after the decode, before writing the file.
    if shared_state_instance.abort_state['level'] >= 2:
        logger.warning(f"Task {task_id}: Hard abort detected before final MP4 write.")
        raise InterruptedError("Hard abort during final save.")

    logger.info(f"Task {task_id}: Writing final MP4 preview...")
    output_queue_ref.push(('progress', (task_id, None, "Writing final MP4 preview...", make_progress_bar_html(100, "Writing MP4..."))))

    final_video_path = os.path.join(outputs_folder, f'{job_id}_aborted_preview.mp4')
    save_bcthw_as_mp4(pixels, final_video_path, fps=fps, crf=crf)
    logger.info(f"Task {task_id}: Saved graceful abort preview to {final_video_path}")
    return final_video_path


def _signal_abort_to_ui(output_queue_ref, task_id, video_path):
    """Helper to send a consistently formatted abort message to the UI queue."""
    logger.info(f"Task {task_id}: Signaling abort to UI, providing video path: {video_path}")
    output_queue_ref.push(('aborted', (task_id, video_path)))