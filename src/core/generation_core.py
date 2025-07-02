﻿import torch
import traceback
import einops
import numpy as np
import os
import json
from PIL import Image
import logging
from PIL.PngImagePlugin import PngInfo

# Local application imports
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import unload_complete_models, load_model_as_complete, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, gpu
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket
from diffusers_helper.gradio.progress_bar import make_progress_bar_html
from core import model_loader # Import model_loader
from ui import metadata as metadata_manager
from ui import shared_state as shared_state_module
from core import generation_utils
from .generation_utils import generate_roll_off_schedule
import traceback
logger = logging.getLogger(__name__)

@torch.no_grad()
def worker(
    # --- Task I/O & Identity ---
    task_id,
    input_image,
    output_folder,
    output_queue_ref,
    # --- Creative Parameters (The "Recipe") ---
    prompt,
    negative_prompt,
    seed,
    video_length,
    steps,
    real_cfg,
    distilled_cfg_start,
    distilled_cfg_end,
    variable_cfg_shape,
    roll_off_start,
    roll_off_factor,
    guidance_rescale,
    preview_frequency,
    segments_to_decode_csv,
    # --- Environment & Debug Parameters ---
    fps,
    latent_window_size,
    gpu_memory_preservation,
    use_teacache,
    use_fp32_transformer_output,
    mp4_crf,
    force_standard_fps,
    # --- Model & System Objects (Passed from main app) ---
    text_encoder,
    text_encoder_2,
    tokenizer,
    tokenizer_2,
    vae,
    feature_extractor,
    image_encoder,
    high_vram,
):
    outputs_folder = os.path.expanduser(output_folder) if output_folder else "./outputs/"
    os.makedirs(outputs_folder, exist_ok=True)

    # --- Job Initialization ---
    total_latent_sections, job_id = generation_utils.initialize_job(
        video_length=video_length,
        fps=fps,
        latent_window_size=latent_window_size,
        task_id=task_id,
        output_queue_ref=output_queue_ref,
    )
    parsed_segments_to_decode_set = set()
    if segments_to_decode_csv:
        try:
            parsed_segments_to_decode_set = {int(s.strip()) for s in segments_to_decode_csv.split(",") if s.strip()}
        except ValueError:
            logger.warning(f"Task {task_id}: Could not parse 'Segments to Decode CSV': \"{segments_to_decode_csv}\".")

    final_output_filename = None
    success = False

    initial_gs_from_ui = distilled_cfg_start
    distilled_cfg_end_value_for_schedule = (
        distilled_cfg_end if distilled_cfg_end is not None else initial_gs_from_ui
    )

    graceful_abort_preview_path = None
    # Ensure transformer is loaded before accessing its properties
    transformer = model_loader.get_transformer_model()
    original_fp32_setting = transformer.high_quality_fp32_output_for_inference # Store original setting

    # For legacy GPUs, we MUST use FP32 output, regardless of the UI setting.
    is_legacy_gpu = shared_state_module.shared_state_instance.system_info.get(shared_state_module.IS_LEGACY_GPU_KEY, False)
    final_use_fp32 = True if is_legacy_gpu else use_fp32_transformer_output
    if is_legacy_gpu and not use_fp32_transformer_output:
        logger.info("Legacy GPU detected: Forcing FP32 transformer output for stability, overriding UI setting.")

    transformer.high_quality_fp32_output_for_inference = final_use_fp32

    # Initialize history_latents_for_abort here to ensure it's always defined
    # It will be overwritten within the loop if generation proceeds
    history_latents_for_abort = None 

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
            "negative_prompt": negative_prompt,
            "seed": seed,
            "video_length": video_length,
            "steps": steps,
            "real_cfg": real_cfg,
            "distilled_cfg_start": distilled_cfg_start,
            "distilled_cfg_end": distilled_cfg_end,
            "guidance_rescale": guidance_rescale,
            "preview_frequency": preview_frequency,
            "segments_to_decode_csv": segments_to_decode_csv,
            "variable_cfg_shape": variable_cfg_shape,
            "roll_off_start": roll_off_start,
            "roll_off_factor": roll_off_factor,
            "fps": fps,
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
            logger.warning(f"Task {task_id}: Failed to save initial image with parameters: {e_png}")

        if not high_vram:
            unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)

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
        if real_cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(
                llama_vec
            ), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(
                negative_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2
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
            # This check for the main interrupt flag makes the Stop button more responsive,
            # allowing it to halt processing between segments.
            if shared_state_module.shared_state_instance.interrupt_flag.is_set():
                logger.info(f"Task {task_id}: Stop signal detected. Breaking generation loop.")
                break
            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size
            # Added for consistent 1-indexed segment number for loop segments
            current_loop_segment_number = latent_padding_iteration + 1
            logger.info(f"Task {task_id}: Seg {current_loop_segment_number}/{total_latent_sections} (lp_val={latent_padding}), last_loop_seg={is_last_section}")

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
                current_diffusion_step = d["i"] + 1
                preview_latent = d["denoised"]
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
                desc = f"Task {task_id}: Vid Frames: {current_video_frames_count}, Len: {current_video_frames_count / fps :.2f}s. Seg {current_loop_segment_number}/{total_latent_sections}. Extending..."
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
                        current_segment_gs_to_use = initial_gs_from_ui + (gs_final_value_for_schedule - initial_gs_from_ui) * curved_progress

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

            # Let the UI know that the expensive VAE decoding is happening
            output_queue_ref.push(
                (
                    "progress",
                    (
                        task_id,
                        None,  # No image preview here
                        f"Segment {current_loop_segment_number}/{total_latent_sections}: Decoding frames...",
                        make_progress_bar_html(
                            100, "VAE Decode"
                        ),  # Progress bar is full from sampling
                    ),
                )
            )

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

            # --- Handle segment saving ---
            saved_file_path = generation_utils.handle_segment_saving(
                latent_padding_iteration=latent_padding_iteration,
                is_last_section=is_last_section,
                current_loop_segment_number=current_loop_segment_number,
                total_latent_sections=total_latent_sections,
                current_video_frame_count=current_video_frame_count,
                history_pixels=history_pixels,
                task_id=task_id,
                job_id=job_id,
                output_queue_ref=output_queue_ref,
                outputs_folder=outputs_folder,
                preview_frequency=preview_frequency,
                parsed_segments_to_decode_set=parsed_segments_to_decode_set,
                fps=fps,
                mp4_crf=mp4_crf,
                playback_compatibility_mode=playback_compatibility_mode,
            )
            if saved_file_path:
                final_output_filename = saved_file_path
                graceful_abort_preview_path = saved_file_path

            history_latents_for_abort = real_history_latents.clone()

        # --- Post-loop logic ---
        success = True

    except (InterruptedError, KeyboardInterrupt) as e:
        logger.info(f"Worker task {task_id} caught explicit abort signal: {e}")
        generation_utils._signal_abort_to_ui(output_queue_ref, task_id, graceful_abort_preview_path)
        success = False
        final_output_filename = graceful_abort_preview_path
    except Exception as e:
        logger.error(f"Error in worker task {task_id}: {e}", exc_info=True)
        output_queue_ref.push(('error', (task_id, str(e))))
        success = False
    finally:
        transformer.high_quality_fp32_output_for_inference = original_fp32_setting
        if not high_vram:
            unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
        output_queue_ref.push(('end', (task_id, success, final_output_filename)))
