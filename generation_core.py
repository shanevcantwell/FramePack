import torch
import traceback
import einops
import numpy as np
import os

from PIL import Image
# from PIL.PngImagePlugin import PngInfo # For PNG metadata later

from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, generate_timestamp
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import unload_complete_models, load_model_as_complete, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, gpu
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket
from diffusers_helper.gradio.progress_bar import make_progress_bar_html


@torch.no_grad()
def worker(
    task_id, input_image_data, prompt, n_prompt, seed, total_second_length,
    latent_window_size, steps, cfg, gs, rs,
    gpu_memory_preservation, use_teacache, mp4_crf, output_folder,
    segments_to_decode_csv_str, # New parameter for selective MP4 saving
    output_queue_ref,
    text_encoder, text_encoder_2, tokenizer, tokenizer_2,
    vae, feature_extractor, image_encoder, transformer,
    high_vram_flag,
    # New parameters for variable GS (Distilled CFG) schedule
    gs_schedule_active=False, 
    gs_final=None # Will default to initial_gs_from_ui if gs_schedule_active is False or gs_final is None
):
    current_output_folder = output_folder if output_folder else './outputs/'
    os.makedirs(current_output_folder, exist_ok=True)

    # Calculate total_latent_sections (total number of segments)
    # This calculation is from demo_gradio_illyasviel.py
    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    job_id = f"{generate_timestamp()}_task{task_id}"
    output_queue_ref.push(('progress', (task_id, None, f'Total Segments: {total_latent_sections}', make_progress_bar_html(0, 'Starting ...'))))

    # Parse segments_to_decode_csv_str
    parsed_segments_to_decode_set = set()
    if segments_to_decode_csv_str:
        try:
            parsed_segments_to_decode_set = {int(s.strip()) for s in segments_to_decode_csv_str.split(',') if s.strip()}
        except ValueError:
            print(f"Task {task_id}: Warning - Could not parse 'Segments to Decode CSV'. Contains non-integer values. Will only save first/last segment.")
            # Optionally send a warning to UI via output_queue_ref
    
    final_output_filename = None
    success = False
    
    # --- Variable GS setup ---
    initial_gs_from_ui = gs # Original gs value from UI
    if gs_final is None: # If gs_final is not provided for schedule, use initial value
        gs_final_value_for_schedule = initial_gs_from_ui
    else:
        gs_final_value_for_schedule = gs_final

    try:
        if not isinstance(input_image_data, np.ndarray):
            raise ValueError(f"Task {task_id}: input_image_data is not a NumPy array. Type: {type(input_image_data)}")

        if not high_vram_flag:
            unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)

        output_queue_ref.push(('progress', (task_id, None, f'Total Segments: {total_latent_sections}', make_progress_bar_html(0, 'Text encoding ...'))))
        if not high_vram_flag:
            fake_diffusers_current_device(text_encoder, gpu)
            load_model_as_complete(text_encoder_2, target_device=gpu)

        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        output_queue_ref.push(('progress', (task_id, None, f'Total Segments: {total_latent_sections}', make_progress_bar_html(0, 'Image processing ...'))))
        current_input_image_np = input_image_data
        if current_input_image_np.shape[-1] == 4:
            pil_img = Image.fromarray(current_input_image_np)
            current_input_image_np = np.array(pil_img.convert("RGB"))
        H, W, C = current_input_image_np.shape
        if C != 3:
             raise ValueError(f"Task {task_id}: Input image must have 3 channels (RGB), found {C}")
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np_resized = resize_and_center_crop(current_input_image_np, target_width=width, target_height=height)
        
        # --- PNG Metadata Saving for initial image ---
        # This logic is from demo_gradio_illyasviel.py
        # (also present in demo_gradio_pr178_png_prompt_metadata_suggested_implementation.py)
        from PIL.PngImagePlugin import PngInfo # Local import for this section
        metadata = PngInfo()
        metadata.add_text("prompt", prompt)
        metadata.add_text("seed", str(seed))
        # Also add other relevant params if desired
        metadata.add_text("total_second_length", str(total_second_length))
        metadata.add_text("steps", str(steps))
        metadata.add_text("gs", str(gs)) # Initial GS
        if gs_schedule_active:
            metadata.add_text("gs_final", str(gs_final_value_for_schedule))
            metadata.add_text("gs_schedule_active", "True")

        input_image_with_metadata_path = os.path.join(current_output_folder, f'{job_id}_input_metadata.png')
        try:
            Image.fromarray(input_image_np_resized).save(input_image_with_metadata_path, pnginfo=metadata)
            print(f"Task {task_id}: Saved input image with metadata to {input_image_with_metadata_path}")
        except Exception as e_png:
            print(f"Task {task_id}: Warning - Failed to save input image with metadata: {e_png}")
        # --- End PNG Metadata Saving ---

        input_image_pt = torch.from_numpy(input_image_np_resized).float() / 127.5 - 1.0
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        output_queue_ref.push(('progress', (task_id, None, f'Total Segments: {total_latent_sections}', make_progress_bar_html(0, 'VAE encoding ...'))))
        if not high_vram_flag:
            load_model_as_complete(vae, target_device=gpu)
        start_latent = vae_encode(input_image_pt.to(vae.device, dtype=vae.dtype), vae)

        output_queue_ref.push(('progress', (task_id, None, f'Total Segments: {total_latent_sections}', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))
        if not high_vram_flag:
            load_model_as_complete(image_encoder, target_device=gpu)
        image_encoder_output = hf_clip_vision_encode(input_image_np_resized, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        output_queue_ref.push(('progress', (task_id, None, f'Total Segments: {total_latent_sections}', make_progress_bar_html(0, 'Start sampling ...'))))
        rnd = torch.Generator(device="cpu").manual_seed(int(seed))
        num_frames_per_segment_generation = latent_window_size * 4 - 3 # (aliased as num_frames)

        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32, device="cpu") #
        history_pixels = None
        total_generated_latent_frames = 0
        
        # Latent padding logic from demo_gradio_illyasviel.py
        latent_paddings_loop_source = list(reversed(range(total_latent_sections)))
        if total_latent_sections > 4:
            latent_paddings_loop_source = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        for latent_padding_iteration, latent_padding_value in enumerate(latent_paddings_loop_source):
            current_segment_0_indexed = latent_padding_iteration # For user display (0 to N-1)
            is_last_section_loop = (latent_padding_value == 0) # True for the very last segment to be processed by worker
            
            # For clarity in soft_append and VAE decode, is_first_segment_processed_by_worker is when latent_padding_iteration == 0
            is_first_segment_processed_by_worker = (latent_padding_iteration == 0)

            latent_padding_size = latent_padding_value * latent_window_size #

            print(f'Task {task_id}: Segment {current_segment_0_indexed + 1}/{total_latent_sections} (latent_padding_value={latent_padding_value}), is_last_section_loop={is_last_section_loop}')

            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16]), device="cpu").unsqueeze(0) #
            clean_latent_indices_pre, _, latent_indices_for_sampling, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1) #
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1) #

            clean_latents_pre = start_latent.to(history_latents.device, dtype=history_latents.dtype)
            
            # Ensure history_latents is deep enough for split
            current_history_depth = history_latents.shape[2]
            needed_depth_for_clean_split = 1 + 2 + 16 
            if current_history_depth < needed_depth_for_clean_split:
                 padding_needed = needed_depth_for_clean_split - current_history_depth
                 pad_tensor = torch.zeros(history_latents.shape[0], history_latents.shape[1], padding_needed, history_latents.shape[3], history_latents.shape[4],
                                          dtype=history_latents.dtype, device=history_latents.device)
                 history_latents_for_clean_split = torch.cat((history_latents, pad_tensor), dim=2)
            else:
                 history_latents_for_clean_split = history_latents
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents_for_clean_split[:, :, :needed_depth_for_clean_split, :, :].split([1, 2, 16], dim=2) #
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2) #

            if not high_vram_flag:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            def callback_diffusion_step(d):
                preview_latent = d['denoised']
                preview_img_np = vae_decode_fake(preview_latent)
                preview_img_np = (preview_img_np * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                preview_img_np = einops.rearrange(preview_img_np, 'b c t h w -> (b h) (t w) c')
                current_diffusion_step = d['i'] + 1
                percentage = int(100.0 * current_diffusion_step / steps)
                hint = f'Segment {current_segment_0_indexed + 1}, Sampling {current_diffusion_step}/{steps}'
                desc = f'Task {task_id}: Vid Len: {max(0, (total_generated_latent_frames * 4 - 3) / 30) :.2f}s. Seg {current_segment_0_indexed + 1}/{total_latent_sections}. Extending...'
                output_queue_ref.push(('progress', (task_id, preview_img_np, desc, make_progress_bar_html(percentage, hint))))
                return

            # --- Variable GS calculation for current segment ---
            current_segment_gs_to_use = initial_gs_from_ui
            if gs_schedule_active and total_latent_sections > 1:
                progress = current_segment_0_indexed / (total_latent_sections - 1)
                current_segment_gs_to_use = initial_gs_from_ui + (gs_final_value_for_schedule - initial_gs_from_ui) * progress
                print(f"Task {task_id}: Segment {current_segment_0_indexed + 1}, Scheduled gs: {current_segment_gs_to_use:.2f}")
            # --- End Variable GS ---

            try:
                 generated_latents_gpu = sample_hunyuan(
                     transformer=transformer, sampler='unipc', width=width, height=height,
                     frames=num_frames_per_segment_generation, real_guidance_scale=cfg,
                     distilled_guidance_scale=current_segment_gs_to_use, # Use scheduled GS
                     guidance_rescale=rs, num_inference_steps=steps, generator=rnd,
                     prompt_embeds=llama_vec.to(transformer.device), prompt_embeds_mask=llama_attention_mask.to(transformer.device),
                     prompt_poolers=clip_l_pooler.to(transformer.device),
                     negative_prompt_embeds=llama_vec_n.to(transformer.device), negative_prompt_embeds_mask=llama_attention_mask_n.to(transformer.device),
                     negative_prompt_poolers=clip_l_pooler_n.to(transformer.device),
                     device=transformer.device, dtype=transformer.dtype, # Use transformer's dtype
                     image_embeddings=image_encoder_last_hidden_state.to(transformer.device),
                     latent_indices=latent_indices_for_sampling.to(transformer.device), # Renamed for clarity
                     clean_latents=clean_latents.to(transformer.device, dtype=transformer.dtype),
                     clean_latent_indices=clean_latent_indices.to(transformer.device),
                     clean_latents_2x=clean_latents_2x.to(transformer.device, dtype=transformer.dtype),
                     clean_latent_2x_indices=clean_latent_2x_indices.to(transformer.device),
                     clean_latents_4x=clean_latents_4x.to(transformer.device, dtype=transformer.dtype),
                     clean_latent_4x_indices=clean_latent_4x_indices.to(transformer.device),
                     callback=callback_diffusion_step,
                 )
            except KeyboardInterrupt:
                 print(f"Task {task_id} received interrupt signal during sampling.")
                 output_queue_ref.push(('aborted', task_id))
                 return

            generated_latents_cpu = generated_latents_gpu.to(dtype=torch.float32, device="cpu") # Move to CPU for append
            
            # If this is the last section to be processed by worker (is_last_section_loop is True),
            # it means this generated_latents_cpu represents the *true beginning* of the video.
            # The start_latent (from initial image) should be appended to it if this is the very end of the entire video generation.
            # The original illyasviel logic: `if is_last_section: generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)`
            # This prepends start_latent to the *output of sample_hunyuan for the final segment*.
            # This means start_latent becomes the *very first frame(s)* of that segment.
            if is_last_section_loop:
                generated_latents_cpu = torch.cat([start_latent.to(device="cpu", dtype=torch.float32), generated_latents_cpu], dim=2)


            # Update total_generated_latent_frames with the number of frames in the current segment
            # This must account for the potentially prepended start_latent in the final segment
            current_segment_num_actual_frames = generated_latents_cpu.shape[2]
            total_generated_latent_frames += current_segment_num_actual_frames

            # Append current segment's latents to the *beginning* of history_latents (since we generate backwards in time)
            # The history_latents grows from the "end" (input image) towards the "start" (generated frames)
            # So, new content is prepended.
            if is_first_segment_processed_by_worker: # For the very first segment generated by worker
                history_latents = generated_latents_cpu # This segment contains the input image (as start_latent was part of its clean_latents)
            else:
                # Prepend the new segment's latents to the existing history_latents
                # Ensure history_latents doesn't overgrow if not needed (though it's implicitly managed by total_latent_sections)
                history_latents = torch.cat([generated_latents_cpu, history_latents], dim=2)

            # Trim history_latents to keep only up to total_generated_latent_frames.
            # This is important if concatenation logic above leads to more frames than expected.
            # However, if total_generated_latent_frames correctly tracks, this might not be strictly necessary
            # if the target size of history_latents is known (e.g., related to total_second_length).
            # For now, assume total_generated_latent_frames is the source of truth for length.
            history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]


            if not high_vram_flag:
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(vae, target_device=gpu)
            
            # --- VAE Decode for soft_append_bcthw (ESSENTIAL, cannot be skipped) ---
            # This decodes only the *current* generated latent segment for stitching
            newly_decoded_pixels = vae_decode(generated_latents_cpu.to(vae.device, dtype=vae.dtype), vae).cpu()
            # ---

            if is_first_segment_processed_by_worker: # This is the first segment processed by the worker.
                 history_pixels = newly_decoded_pixels # It becomes the initial pixel history.
            else:
                 # overlapped_frames is the number of frames in each generated segment by sample_hunyuan
                 overlapped_frames = num_frames_per_segment_generation 
                 history_pixels = soft_append_bcthw(newly_decoded_pixels, history_pixels, overlap=overlapped_frames)


            if not high_vram_flag:
                unload_complete_models(vae)
            
            # --- Conditional MP4 Save ---
            should_save_mp4_this_iteration = False
            # Segment numbers for user are 1-indexed
            current_segment_1_indexed = current_segment_0_indexed + 1

            if parsed_segments_to_decode_set: # User provided a list
                if current_segment_1_indexed in parsed_segments_to_decode_set:
                    should_save_mp4_this_iteration = True
            # If CSV is empty, only mandatory first/last saves apply by default.
            
            if is_first_segment_processed_by_worker: # Always save MP4 for the first processed segment
                should_save_mp4_this_iteration = True
            
            if is_last_section_loop: # Always save MP4 for the final composite video
                should_save_mp4_this_iteration = True

            if should_save_mp4_this_iteration:
                segment_mp4_filename = os.path.join(current_output_folder, f'{job_id}_segment_{current_segment_1_indexed}_totalframes_{history_pixels.shape[2]}.mp4')
                save_bcthw_as_mp4(history_pixels, segment_mp4_filename, fps=30, crf=mp4_crf)
                final_output_filename = segment_mp4_filename # Update with the latest saved MP4
                print(f"Task {task_id}: SAVED MP4 for segment {current_segment_1_indexed} to {segment_mp4_filename}. Pixel history frames: {history_pixels.shape[2]}")
                output_queue_ref.push(('file', (task_id, segment_mp4_filename, f"Segment {current_segment_1_indexed} saved"))) # Send file info
            else:
                print(f"Task {task_id}: SKIPPED MP4 save for segment {current_segment_1_indexed}.")
            # --- End Conditional MP4 Save ---

            if is_last_section_loop: # This was the final segment in the padding loop
                success = True
                break # Exit the main generation loop
        
        if not high_vram_flag: # Final unload
            unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)

    except KeyboardInterrupt:
        print(f"Worker for task {task_id} caught KeyboardInterrupt.")
        output_queue_ref.push(('aborted', task_id))
        success = False
    except Exception as e:
        print(f"Error in worker for task {task_id}:")
        traceback.print_exc()
        output_queue_ref.push(('error', (task_id, str(e))))
        success = False
    finally:
        if not high_vram_flag:
            unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
        
        if final_output_filename and not os.path.dirname(final_output_filename) == os.path.abspath(current_output_folder):
             final_output_filename = os.path.join(current_output_folder, os.path.basename(final_output_filename))
        
        output_queue_ref.push(('end', (task_id, success, final_output_filename)))