import torch
import traceback
import einops
import numpy as np
import os
import json
from PIL import Image
from PIL.PngImagePlugin import PngInfo

# Local application imports
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, generate_timestamp
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import unload_complete_models, load_model_as_complete, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, gpu
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket
from diffusers_helper.gradio.progress_bar import make_progress_bar_html
from ui import metadata as metadata_manager
from ui import shared_state
from core import generation_utils
from .generation_utils import generate_roll_off_schedule

import traceback

def _save_final_preview(history_latents, vae, job_id, task_id, outputs_folder, crf, output_queue_ref, high_vram):
    """
    Helper function to decode and save the final video preview during a graceful abort.
    This logic is refactored to be callable from the end of the worker.
    CHANGED: Corrected parameter name from `history_pixels` to `history_latents` for accuracy.
    """
    if history_latents is None:
        print(f"Task {task_id}: No latents generated, cannot save final preview.")
        return None

    # CHANGED: Added check for a hard abort (level 2) before starting the expensive decode.
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

    # CHANGED: Added a second check for a hard abort after the decode, before writing the file.
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


@torch.no_grad()
def worker(
    # --- Task I/O & Identity ---
    task_id,
    input_image,
    output_folder,
    output_queue_ref,
    # --- Creative Parameters (The "Recipe") ---
    prompt,
    n_prompt,
    seed,
    total_second_length,
    steps,
    cfg,
    gs,
    gs_final,
    # --- MODIFIED: Parameters for the new CFG scheduler ---
    gs_schedule_shape,
    roll_off_start,
    roll_off_factor,
    # --- End of modified parameters ---
    rs,
    preview_frequency,
    segments_to_decode_csv,
    # --- Environment & Debug Parameters ---
    latent_window_size,
    gpu_memory_preservation,
    use_teacache,
    use_fp32_transformer_output,
    mp4_crf,
    # --- Model & System Objects (Passed from main app) ---
    text_encoder,
    text_encoder_2,
    tokenizer,
    tokenizer_2,
    vae,
    feature_extractor,
    image_encoder,
    transformer,
    high_vram,
):
    outputs_folder = os.path.expanduser(output_folder) if output_folder else "./outputs/"
    os.makedirs(outputs_folder, exist_ok=True)

    # --- Gemini: do not touch - "secret sauce"
    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

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
    # ---
    parsed_segments_to_decode_set = set()
    if segments_to_decode_csv:
        try:
            parsed_segments_to_decode_set = {int(s.strip()) for s in segments_to_decode_csv.split(",") if s.strip()}
        except ValueError:
            print(f"Task {task_id}: Warning - Could not parse 'Segments to Decode CSV': \"{segments_to_decode_csv}\".")

    final_output_filename = None
    success = False

    initial_gs_from_ui = gs
    gs_final_value_for_schedule = (
        gs_final if gs_final is not None else initial_gs_from_ui
    )

    graceful_abort_preview_path = None
    original_fp32_setting = transformer.high_quality_fp32_output_for_inference
    transformer.high_quality_fp32_output_for_inference = use_fp32_transformer_output

    # Initialize history_latents_for_abort here to ensure it's always defined
    # It will be overwritten within the loop if generation proceeds
    history_latents_for_abort = None # Added: Initialize history_latents_for_abort to None

    try:
        if not isinstance(input_image, np.ndarray):
            raise ValueError(f"Task {task_id}: input_image is not a NumPy array.")
        output_queue_ref.push(('progress', (task_id, None, f'Total Segments: {total_latent_sections}', make_progress_bar_html(0, "Image processing ..."))))
        if input_image.shape[-1] == 4:
            pil_img = Image.fromarray(input_image)
            input_image = np.array(pil_img.convert("RGB"))
        H, W, C = input_image.shape
        if C != 3: raise ValueError(f"Task {task_id}: Input image must be RGB, found {C} channels.")
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)

        metadata_obj = PngInfo()
        params_to_save_in_metadata = {
            "prompt": prompt,
            "n_prompt": n_prompt,
            "seed": seed,
            "total_second_length": total_second_length,
            "steps": steps,
            "cfg": cfg,
            "gs": gs,
            "gs_final": gs_final,
            "rs": rs,
            "preview_frequency": preview_frequency,
            "segments_to_decode_csv": segments_to_decode_csv,
            "gs_schedule_shape": gs_schedule_shape,
            "roll_off_start": roll_off_start,
            "roll_off_factor": roll_off_factor,
        }
        metadata_obj.add_text("parameters", json.dumps(params_to_save_in_metadata))
        initial_image_with_params_path = os.path.join(
            outputs_folder, f"{job_id}_initial_image_with_params.png"
        )
        try:
            Image.fromarray(input_image_np).save(
                initial_image_with_params_path, pnginfo=metadata_obj
            )
        except Exception as e_png:
            print(
                f"Task {task_id}: WARNING - Failed to save initial image with parameters: {e_png}"
            )

        # --- Gemini: do not touch - "secret sauce"
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )
        output_queue_ref.push(
            (
                "progress",
                (
                    task_id,
                    None,
                    f"Total Segments: {total_latent_sections}",
                    make_progress_bar_html(0, "Text encoding ..."),
                ),
            )
        )
        if not high_vram:
            fake_diffusers_current_device(text_encoder, gpu)
            load_model_as_complete(text_encoder_2, target_device=gpu)
        llama_vec, clip_l_pooler = encode_prompt_conds(
            prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2
        )
        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(
                llama_vec
            ), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(
                n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2
            )
        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(
            llama_vec_n, length=512
        )
        input_image_pt = (
            torch.from_numpy(input_image_np).float().permute(2, 0, 1).unsqueeze(0)
            / 127.5
            - 1.0
        )
        input_image_pt = input_image_pt[:, :, None, :, :]
        output_queue_ref.push(
            (
                "progress",
                (
                    task_id,
                    None,
                    f"Total Segments: {total_latent_sections}",
                    make_progress_bar_html(0, "VAE encoding ..."),
                ),
            )
        )
        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)
        start_latent = vae_encode(input_image_pt, vae)
        output_queue_ref.push(
            (
                "progress",
                (
                    task_id,
                    None,
                    f"Total Segments: {total_latent_sections}",
                    make_progress_bar_html(0, "CLIP Vision encoding ..."),
                ),
            )
        )
        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)
        image_encoder_output = hf_clip_vision_encode(
            input_image_np, feature_extractor, image_encoder
        )
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
        (
            llama_vec,
            llama_vec_n,
            clip_l_pooler,
            clip_l_pooler_n,
            image_encoder_last_hidden_state,
        ) = [
            t.to(transformer.dtype)
            for t in [
                llama_vec,
                llama_vec_n,
                clip_l_pooler,
                clip_l_pooler_n,
                image_encoder_last_hidden_state,
            ]
        ]

        output_queue_ref.push(
            (
                "progress",
                (
                    task_id,
                    None,
                    f"Total Segments: {total_latent_sections}",
                    make_progress_bar_html(0, "Start sampling ..."),
                ),
            )
        )
        rnd = torch.Generator(device="cpu").manual_seed(int(seed))
        num_frames = latent_window_size * 4 - 3

        history_latents = torch.zeros(
            size=(1, 16, 1 + 2 + 16, height // 8, width // 8),
            dtype=torch.float32,
            device="cpu",
        )
        history_pixels = None
        total_generated_latent_frames = 0
        latent_paddings = list(reversed(range(total_latent_sections)))
        if total_latent_sections > 4:
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        for latent_padding_iteration, latent_padding in enumerate(latent_paddings):
            # CHANGED: This check handles a graceful abort (level 1+). A single click
            # will break this loop, and execution will jump to the post-loop logic.
            if shared_state.abort_state['level'] >= 1:
                print(f"Task {task_id}: Graceful abort detected. Breaking generation loop to save final preview.")
                break
            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size
            # Added for consistent 1-indexed segment number for loop segments
            current_loop_segment_number = latent_padding_iteration + 1
            print(
                f"Task {task_id}: Seg {current_loop_segment_number}/{total_latent_sections} (lp_val={latent_padding}), last_loop_seg={is_last_section}"
            )

            indices = torch.arange(
                0,
                sum([1, latent_padding_size, latent_window_size, 1, 2, 16]),
                device="cpu",
            ).unsqueeze(0)
            (
                clean_latent_indices_pre,
                _,
                latent_indices,
                clean_latent_indices_post,
                clean_latent_2x_indices,
                clean_latent_4x_indices,
            ) = indices.split(
                [1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1
            )
            clean_latents_pre = start_latent.to(
                history_latents.device, dtype=history_latents.dtype
            )
            clean_latent_indices = torch.cat(
                [clean_latent_indices_pre, clean_latent_indices_post], dim=1
            )

            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[
                :, :, : 1 + 2 + 16, :, :
            ].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            if not high_vram:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(
                    transformer,
                    target_device=gpu,
                    preserved_memory_gb=gpu_memory_preservation,
                )
            transformer.initialize_teacache(
                enable_teacache=use_teacache, num_steps=steps
            )

            def callback_diffusion_step(d):
                if shared_state.abort_state['level'] >= 2:
                    raise KeyboardInterrupt("Abort signal received during sampling.")
                
                # MODIFIED: Removed the conditional return to ensure the UI is updated on every step.
                current_diffusion_step = d["i"] + 1

                preview_latent = d["denoised"]
                # ... (rest of the image processing) ...
                preview_img_np = vae_decode_fake(preview_latent)
                preview_img_np = (
                    (preview_img_np * 255.0)
                    .detach()
                    .cpu()
                    .numpy()
                    .clip(0, 255)
                    .astype(np.uint8)
                )
                preview_img_np = einops.rearrange(
                    preview_img_np, "b c t h w -> (b h) (t w) c"
                )

                percentage = int(100.0 * current_diffusion_step / steps)
                hint = f"Segment {current_loop_segment_number}, Sampling {current_diffusion_step}/{steps}"
                current_video_frames_count = (
                    history_pixels.shape[2] if history_pixels is not None else 0
                )
                desc = f"Task {task_id}: Vid Frames: {current_video_frames_count}, Len: {current_video_frames_count / 30 :.2f}s. Seg {current_loop_segment_number}/{total_latent_sections}. Extending..."
                output_queue_ref.push(
                    (
                        "progress",
                        (
                            task_id,
                            preview_img_np,
                            desc,
                            make_progress_bar_html(percentage, hint),
                        ),
                    )
                )

            current_segment_gs_to_use = initial_gs_from_ui
            # Only apply a schedule if one is selected and there's more than one segment.
            if gs_schedule_shape != 'Off' and total_latent_sections > 1:
                # Calculate progress as a value from 0.0 to 1.0 over the segments.
                progress = latent_padding_iteration / (total_latent_sections - 1)

                if gs_schedule_shape == 'Linear':
                    # Linear interpolation from start to end CFG.
                    current_segment_gs_to_use = initial_gs_from_ui + (gs_final_value_for_schedule - initial_gs_from_ui) * progress
               
                elif gs_schedule_shape == 'Roll-off':
                    # Roll-off logic adapted for per-segment scheduling.
                    roll_off_start_point = roll_off_start / 100.0
                    if progress < roll_off_start_point:
                        current_segment_gs_to_use = initial_gs_from_ui
                    else:
                        roll_off_progress = (progress - roll_off_start_point) / (1.0 - roll_off_start_point)
                        curved_progress = roll_off_progress ** roll_off_factor
                        current_segment_gs_to_use = initial_gs_from_ui (gs_final_value_for_schedule - initial_gs_from_ui) * curved_progress


                generated_latents = sample_hunyuan(
                    transformer=transformer,
                    sampler="unipc",
                    width=width,
                    height=height,
                    frames=num_frames,
                    real_guidance_scale=cfg,
                    distilled_guidance_scale=current_segment_gs_to_use,
                    guidance_rescale=rs,
                    num_inference_steps=steps,
                    generator=rnd,
                    prompt_embeds=llama_vec.to(transformer.device),
                    prompt_embeds_mask=llama_attention_mask.to(transformer.device),
                    prompt_poolers=clip_l_pooler.to(transformer.device),
                    negative_prompt_embeds=llama_vec_n.to(transformer.device),
                    negative_prompt_embeds_mask=llama_attention_mask_n.to(
                        transformer.device
                    ),
                    negative_prompt_poolers=clip_l_pooler_n.to(transformer.device),
                    device=transformer.device,
                    dtype=transformer.dtype,
                    image_embeddings=image_encoder_last_hidden_state.to(transformer.device),
                    latent_indices=latent_indices.to(transformer.device),
                    clean_latents=clean_latents.to(
                        transformer.device, dtype=transformer.dtype
                    ),
                    clean_latent_indices=clean_latent_indices.to(transformer.device),
                    clean_latents_2x=clean_latents_2x.to(
                        transformer.device, dtype=transformer.dtype
                    ),
                    clean_latent_2x_indices=clean_latent_2x_indices.to(transformer.device),
                    clean_latents_4x=clean_latents_4x.to(
                        transformer.device, dtype=transformer.dtype
                    ),
                    clean_latent_4x_indices=clean_latent_4x_indices.to(transformer.device),
                    callback=callback_diffusion_step,
                )

                if is_last_section:
                    generated_latents = torch.cat(
                        [start_latent.to(generated_latents), generated_latents], dim=2
                    )

                total_generated_latent_frames += int(generated_latents.shape[2])
                history_latents = torch.cat(
                    [generated_latents.to(history_latents), history_latents], dim=2
                )

                if not high_vram:
                    offload_model_from_device_for_memory_preservation(
                        transformer, target_device=gpu, preserved_memory_gb=8
                    )
                    load_model_as_complete(vae, target_device=gpu)

                real_history_latents = history_latents[
                    :, :, :total_generated_latent_frames, :, :
                ]

                if history_pixels is None:
                    history_pixels = vae_decode(real_history_latents, vae).cpu()
                else:
                    section_latent_frames = (
                        (latent_window_size * 2 + 1)
                        if is_last_section
                        else (latent_window_size * 2)
                    )
                    overlapped_frames = latent_window_size * 4 - 3
                    current_pixels = vae_decode(
                        real_history_latents[:, :, :section_latent_frames], vae
                    ).cpu()
                    history_pixels = soft_append_bcthw(
                        current_pixels, history_pixels, overlapped_frames
                    )

                if not high_vram:
                    unload_complete_models()

                current_video_frame_count = history_pixels.shape[2]

                # --- Gemini start again
                # --- Unified and Corrected MP4 Saving Logic ---
                should_save_mp4_this_iteration = False

                # Condition 1: Always save the very first segment (iteration 0)
                if latent_padding_iteration == 0:
                    should_save_mp4_this_iteration = True
                # Condition 2: Always save the last segment
                elif is_last_section:
                    should_save_mp4_this_iteration = True
                # Condition 3: Save if the user specified this segment number (1-indexed)
                elif (
                    parsed_segments_to_decode_set
                    and current_loop_segment_number in parsed_segments_to_decode_set
                ):
                    should_save_mp4_this_iteration = True
                # Condition 4: Save based on the periodic preview_frequency setting
                elif (
                    preview_frequency > 0
                    and (current_loop_segment_number % preview_frequency == 0)
                ):
                    should_save_mp4_this_iteration = True

                if should_save_mp4_this_iteration:
                    segment_mp4_filename = os.path.join(
                        outputs_folder,
                        f"{job_id}_segment_{current_loop_segment_number}_frames_{current_video_frame_count}.mp4",
                    )
                    save_bcthw_as_mp4(
                        history_pixels, segment_mp4_filename, fps=30, crf=mp4_crf
                    )
                    final_output_filename = segment_mp4_filename
                    print(
                        f"Task {task_id}: SAVED MP4 for segment {current_loop_segment_number} to {segment_mp4_filename}. Total video frames: {current_video_frame_count}"
                    )
                    output_queue_ref.push(
                        (
                            "file",
                            (
                                task_id,
                                segment_mp4_filename,
                                f"Segment {current_loop_segment_number} MP4 saved ({current_video_frame_count} frames)",
                            ),
                        )
                    )
                else:
                    print(
                        f"Task {task_id}: SKIPPED MP4 save for intermediate segment {current_loop_segment_number}."
                    )
                history_latents_for_abort = real_history_latents.clone()

        # --- Post-loop logic ---
        # If the loop was broken by a graceful abort (level 1), save the preview.
        if shared_state.abort_state['level'] == 1:
            graceful_abort_preview_path = generation_utils._save_final_preview(                history_latents_for_abort, vae, job_id, task_id, outputs_folder, mp4_crf, output_queue_ref, high_vram
            )
            # A graceful abort is not a full success, but we provide the preview path.
            success = False
            final_output_filename = graceful_abort_preview_path
            generation_utils._signal_abort_to_ui(output_queue_ref, task_id, graceful_abort_preview_path)
        # This else block runs only if the loop completed naturally without an abort signal.
        else:
            success = True
            # The final filename was already set by the last iteration of the loop.

    except (InterruptedError, KeyboardInterrupt) as e:
        print(f"Worker task {task_id} caught explicit abort signal: {e}")
        generation_utils._signal_abort_to_ui(output_queue_ref, task_id, graceful_abort_preview_path)
        success = False
        final_output_filename = graceful_abort_preview_path
    except Exception as e:
        print(f"Error in worker task {task_id}: {e}")
        traceback.print_exc()
        output_queue_ref.push(('error', (task_id, str(e))))
        success = False
    finally:
        # This block now correctly reports the outcome
        transformer.high_quality_fp32_output_for_inference = original_fp32_setting
        if not high_vram:
            unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
        output_queue_ref.push(('end', (task_id, success, final_output_filename)))
