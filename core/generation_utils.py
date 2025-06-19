# core/generation_utils.py
# NEW FILE: Contains helper functions refactored from generation_core.py

import os
from ui import shared_state
from diffusers_helper.memory import load_model_as_complete, unload_complete_models, gpu
from diffusers_helper.hunyuan import vae_decode
from diffusers_helper.utils import save_bcthw_as_mp4
from diffusers_helper.gradio.progress_bar import make_progress_bar_html
from . import generation_utils

def _save_final_preview(history_latents, vae, job_id, task_id, outputs_folder, crf, output_queue_ref, high_vram):
    """
    Helper function to decode and save the final video preview during a graceful abort.
    This logic is refactored to be callable from the end of the worker.
    """
    if history_latents is None:
        print(f"Task {task_id}: No latents generated, cannot save final preview.")
        return None

    # Check for a hard abort (level 2) before starting the expensive decode.
    # This allows a double-click abort to interrupt the graceful save.
    if shared_state.abort_state['level'] >= 2:
        print(f"Task {task_id}: Hard abort detected before final VAE decode.")
        raise InterruptedError("Hard abort during final save.")

    print(f"Task {task_id}: Decoding final latents for graceful abort preview...")
    output_queue_ref.push(('progress', (task_id, None, "Decoding final latents for preview...", make_progress_bar_html(100, "Decoding..."))))

    if not high_vram:
        load_model_as_complete(vae, target_device=gpu)

    # This is a blocking, expensive operation.
    pixels = vae_decode(history_latents, vae).cpu()

    if not high_vram:
        unload_complete_models(vae)

    # Add a second check for a hard abort after the decode, before writing the file.
    if shared_state.abort_state['level'] >= 2:
        print(f"Task {task_id}: Hard abort detected before final MP4 write.")
        raise InterruptedError("Hard abort during final save.")

    print(f"Task {task_id}: Writing final MP4 preview...")
    output_queue_ref.push(('progress', (task_id, None, "Writing final MP4 preview...", make_progress_bar_html(100, "Writing MP4..."))))

    final_video_path = os.path.join(outputs_folder, f'{job_id}_aborted_preview.mp4')
    save_bcthw_as_mp4(pixels, final_video_path, fps=30, crf=crf)
    print(f"Task {task_id}: Saved graceful abort preview to {final_video_path}")
    return final_video_path


def _signal_abort_to_ui(output_queue_ref, task_id, video_path):
    """Helper to send a consistently formatted abort message to the UI queue."""
    print(f"Task {task_id}: Signaling abort to UI, providing video path: {video_path}")
    output_queue_ref.push(('aborted', (task_id, video_path)))