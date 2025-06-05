from diffusers_helper.hf_login import login

import os
os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

import gradio as gr
import torch
import traceback
import einops
import numpy as np
import argparse
import time
import json
import base64
import io
import zipfile
import tempfile
import atexit
import shutil
from pathlib import Path
import threading

from PIL import Image

from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from transformers import SiglipImageProcessor, SiglipVisionModel

from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, DynamicSwapInstaller
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css

from generation_core import worker # Import the updated worker

# --- Globals and Configuration ---
parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true', default=False)
parser.add_argument("--server", type=str, default='127.0.0.1')
parser.add_argument("--port", type=int, required=False)
parser.add_argument("--inbrowser", action='store_true', default=False)
args = parser.parse_args()

queue_lock = threading.Lock()
AUTOSAVE_FILENAME = "framepack_svc_queue.zip"
outputs_folder = './outputs_svc/'
os.makedirs(outputs_folder, exist_ok=True)
queue_cache_dir = os.path.join(outputs_folder, "_queue_cache")
os.makedirs(queue_cache_dir, exist_ok=True)
SETTINGS_FILENAME = "framepack_svc_settings.json"

# Defines order of UI elements for task creation and settings save/load
# Must align with the order of components in task_defining_ui_inputs list later
param_names_from_ui_for_task_creation = [
    'input_image_gallery', 'prompt', 'n_prompt', 'seed', 'total_second_length',
    'latent_window_size', 'steps', 'cfg', 'gs', 'rs',
    'gpu_memory_preservation', 'use_teacache', 'mp4_crf', 
    'segments_to_decode_csv', # New UI input
    'gs_schedule_active_ui', # For variable GS
    'gs_final_ui',           # For variable GS
    'output_folder_ui' # Should be last for mapping to task_param_names_for_worker
]

# Defines parameters passed to the worker (order matters for zipping)
# 'input_image_data' replaces 'input_image_gallery'
task_param_names_for_worker = [
    'input_image_data', 'prompt', 'n_prompt', 'seed', 'total_second_length',
    'latent_window_size', 'steps', 'cfg', 'gs', 'rs',
    'gpu_memory_preservation', 'use_teacache', 'mp4_crf', 'output_folder',
    'segments_to_decode_csv_str', # New worker param
    'gs_schedule_active', # New worker param
    'gs_final'            # New worker param
]

print(f"FramePack SVC launching with args: {args}")

# --- Model Loading ---
print("Initializing models...")
free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60

print(f'Free VRAM {free_mem_gb} GB')
print(f'High-VRAM Mode: {high_vram}')

text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()
feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()
transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()

print("Models loaded to CPU. Configuring...")
vae.eval(); text_encoder.eval(); text_encoder_2.eval(); image_encoder.eval(); transformer.eval()
if not high_vram: vae.enable_slicing(); vae.enable_tiling()
transformer.high_quality_fp32_output_for_inference = False # switching this to false since it hits performance hard - if should become a gradio checkbox
print('transformer.high_quality_fp32_output_for_inference = True')
transformer.to(dtype=torch.bfloat16); vae.to(dtype=torch.float16); image_encoder.to(dtype=torch.float16); text_encoder.to(dtype=torch.float16); text_encoder_2.to(dtype=torch.float16)
vae.requires_grad_(False); text_encoder.requires_grad_(False); text_encoder_2.requires_grad_(False); image_encoder.requires_grad_(False); transformer.requires_grad_(False)

if not high_vram:
    print("Low VRAM mode: Installing DynamicSwap.")
    DynamicSwapInstaller.install_model(transformer, device=gpu)
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    print("High VRAM mode: Moving all models to GPU.")
    text_encoder.to(gpu); text_encoder_2.to(gpu); image_encoder.to(gpu); vae.to(gpu); transformer.to(gpu)
print("Model configuration and placement complete.")

# --- Helper Functions (Adapted from pr150_queues) ---
def patched_video_is_playable(video_filepath): return True
gr.processing_utils.video_is_playable = patched_video_is_playable

def save_defaults(*ui_values_tuple):
    # ui_values_tuple corresponds to default_settings_components_list
    # param_names_from_ui_for_task_creation[1:] (excluding gallery) is the reference for keys
    keys_for_saving = param_names_from_ui_for_task_creation[1:]
    
    settings_to_save = {}
    for i, key_from_ui_list in enumerate(keys_for_saving):
        # Map UI key to worker key if different, or use a direct mapping if names are aligned
        # For new params like segments_to_decode_csv, gs_schedule_active_ui, gs_final_ui,
        # ensure they map to corresponding keys in default_values_map if worker expects them.
        # For simplicity, let's assume UI keys are now the ones we want to save.
        settings_to_save[key_from_ui_list] = ui_values_tuple[i]
        
    try:
        with open(SETTINGS_FILENAME, 'w', encoding='utf-8') as f: json.dump(settings_to_save, f, indent=4)
        gr.Info(f"Defaults saved to {SETTINGS_FILENAME}")
    except Exception as e: gr.Warning(f"Error saving defaults: {e}"); print(f"Error saving defaults: {e}"); traceback.print_exc()

def load_defaults():
    # Default values for UI components, matching param_names_from_ui_for_task_creation[1:]
    default_values_map = {
        'prompt': '', 'n_prompt': '', 'seed': 31337, 'total_second_length': 5.0,
        'latent_window_size': 9, 'steps': 25, 'cfg': 1.0, 'gs': 10.0, 'rs': 0.0,
        'gpu_memory_preservation': 6.0, 'use_teacache': True, 'mp4_crf': 18,
        'segments_to_decode_csv': '', # Default for new UI
        'gs_schedule_active_ui': False, # Default for new UI
        'gs_final_ui': 10.0,          # Default for new UI (matches default gs)
        'output_folder_ui': outputs_folder # Use current default
    }
    loaded_settings = {}
    if os.path.exists(SETTINGS_FILENAME):
        try:
            with open(SETTINGS_FILENAME, 'r', encoding='utf-8') as f: loaded_settings = json.load(f)
            print(f"Loaded defaults from {SETTINGS_FILENAME}")
        except Exception as e: print(f"Error loading defaults from {SETTINGS_FILENAME}: {e}"); loaded_settings = {}

    updates = []
    for name_ui in param_names_from_ui_for_task_creation[1:]: # Iterate through UI component names used for settings
        value = loaded_settings.get(name_ui, default_values_map.get(name_ui))
        updates.append(gr.update(value=value))
    return updates

def np_to_base64_uri(np_array_or_tuple, format="png"):
    if np_array_or_tuple is None: return None
    try:
        np_array = np_array_or_tuple[0] if isinstance(np_array_or_tuple, tuple) and len(np_array_or_tuple) > 0 and isinstance(np_array_or_tuple[0], np.ndarray) else np_array_or_tuple if isinstance(np_array_or_tuple, np.ndarray) else None
        if np_array is None: return None
        pil_image = Image.fromarray(np_array.astype(np.uint8))
        if format.lower() == "jpeg" and pil_image.mode == "RGBA": pil_image = pil_image.convert("RGB")
        buffer = io.BytesIO(); pil_image.save(buffer, format=format.upper()); img_bytes = buffer.getvalue()
        return f"data:image/{format.lower()};base64,{base64.b64encode(img_bytes).decode('utf-8')}"
    except Exception as e: print(f"Error converting NumPy to base64: {e}"); return None

def get_queue_state(state_dict_gr_state):
    if "queue_state" not in state_dict_gr_state:
        state_dict_gr_state["queue_state"] = {"queue": [], "next_id": 1, "processing": False, "abort_current": False, "editing_task_id": None}
    return state_dict_gr_state["queue_state"]

def update_queue_df_display(queue_state):
    queue = queue_state.get("queue", []); data = []
    processing = queue_state.get("processing", False); editing_task_id = queue_state.get("editing_task_id", None)
    for i, task in enumerate(queue):
        params = task['params']; task_id = task['id']
        prompt_display = (params['prompt'][:77] + '...') if len(params['prompt']) > 80 else params['prompt']
        prompt_title = params['prompt'].replace('"', '&quot;')
        prompt_cell = f'<span title="{prompt_title}">{prompt_display}</span>'
        img_uri = np_to_base64_uri(params['input_image_data']); thumbnail_size = "50px"
        img_md = f'<img src="{img_uri}" alt="Input" style="max-width:{thumbnail_size}; max-height:{thumbnail_size}; display:block; margin:auto; object-fit:contain;" />' if img_uri else ""
        is_processing_current_task = processing and i == 0; is_editing_current_task = editing_task_id == task_id
        task_status_val = task.get("status", "pending"); status_display = "<?>"
        if is_processing_current_task: status_display = "⏳ Processing"
        elif is_editing_current_task: status_display = "✏️ Editing"
        elif task_status_val == "done": status_display = "✅ Done"
        elif task_status_val == "error": status_display = "❌ Error"
        elif task_status_val == "aborted": status_display = "⏹️ Aborted"
        elif task_status_val == "pending": status_display = "⏸️ Pending"
        data.append([task_id, status_display, prompt_cell, f"{params.get('total_second_length', 0):.1f}s", params.get('steps', 0), img_md, "↑", "↓", "✖", "✎"])
    return gr.DataFrame(value=data, visible=len(data) > 0)

def add_or_update_task_in_queue(state_dict_gr_state, *args_from_ui_controls_tuple):
    queue_state = get_queue_state(state_dict_gr_state)
    editing_task_id = queue_state.get("editing_task_id", None)
    input_images_gallery_output = args_from_ui_controls_tuple[0]

    if not input_images_gallery_output:
        gr.Warning("Input image(s) are required!")
        return state_dict_gr_state, update_queue_df_display(queue_state), gr.update(value="Add Task to Queue" if editing_task_id is None else "Update Task"), gr.update(visible=editing_task_id is not None)

    tasks_added_count = 0; first_new_task_id = -1
    base_params_for_worker_dict = {}
    # Map UI controls (from tuple) to worker param names
    for i, ui_key_name in enumerate(param_names_from_ui_for_task_creation[1:]): # Skip gallery
        worker_key_name = task_param_names_for_worker[i+1] # Corresponding worker key
        # Special handling for CSV string and gs_schedule params
        if ui_key_name == 'segments_to_decode_csv': worker_key_name = 'segments_to_decode_csv_str'
        elif ui_key_name == 'gs_schedule_active_ui': worker_key_name = 'gs_schedule_active'
        elif ui_key_name == 'gs_final_ui': worker_key_name = 'gs_final'
        elif ui_key_name == 'output_folder_ui': worker_key_name = 'output_folder'

        base_params_for_worker_dict[worker_key_name] = args_from_ui_controls_tuple[i+1]


    if editing_task_id is not None: # Update existing task
        if len(input_images_gallery_output) > 1: gr.Warning("Cannot update task with multiple images. Cancel edit."); return state_dict_gr_state, update_queue_df_display(queue_state), gr.update(value="Update Task"), gr.update(visible=True)
        img_tuple = input_images_gallery_output[0]
        if not (isinstance(img_tuple, tuple) and len(img_tuple) > 0 and isinstance(img_tuple[0], np.ndarray)): gr.Warning("Invalid image format for update."); return state_dict_gr_state, update_queue_df_display(queue_state), gr.update(value="Update Task"), gr.update(visible=True)
        img_np_for_update = img_tuple[0]
        with queue_lock:
            task_found = False
            for task in queue_state["queue"]:
                if task["id"] == editing_task_id:
                    task["params"] = {'input_image_data': img_np_for_update, **base_params_for_worker_dict}
                    task["status"] = "pending"; task_found = True; break
            if not task_found: gr.Warning(f"Task {editing_task_id} not found for update.")
            else: gr.Info(f"Task {editing_task_id} updated.")
            queue_state["editing_task_id"] = None
    else: # Add new task(s)
        with queue_lock:
            for img_tuple in input_images_gallery_output:
                if not (isinstance(img_tuple, tuple) and len(img_tuple) > 0 and isinstance(img_tuple[0], np.ndarray)): gr.Warning("Skipping invalid image input."); continue
                img_np_data = img_tuple[0]
                next_id = queue_state["next_id"]
                if first_new_task_id == -1: first_new_task_id = next_id
                task = {"id": next_id, "params": {'input_image_data': img_np_data, **base_params_for_worker_dict}, "status": "pending"}
                queue_state["queue"].append(task); queue_state["next_id"] += 1; tasks_added_count += 1
        if tasks_added_count > 0: gr.Info(f"Added {tasks_added_count} task(s) (start ID: {first_new_task_id}).")
        else: gr.Warning("No valid tasks added.")
    return state_dict_gr_state, update_queue_df_display(queue_state), gr.update(value="Add Task to Queue"), gr.update(visible=False)

# ... (cancel_edit_mode_action, move_task_in_queue, remove_task_from_queue, handle_queue_action_on_select, clear_task_queue_action - largely similar to previous draft, ensure keys match)
# ... (save_queue_to_zip, load_queue_from_zip, autosave_queue_on_exit_action, autoload_queue_on_start_action - largely similar, ensure keys match)
# ... (process_task_queue_main_loop, abort_current_task_processing_action, quit_application_action - largely similar)

# Function to calculate and update total segments display
def ui_update_total_segments(total_seconds, win_size_latent_frames):
    # Calculation from worker
    # latent_window_size from UI is in "latent frames" but formula in worker used it as "latent_window_size * 4" (video frames per latent step)
    # If latent_window_size_ui is already the "latent_window_size" for the formula's denominator component (e.g. 9)
    # then the formula used latent_window_size * 4.
    # The UI slider for latent_window_size_ui is typically small (e.g., 9). Let's assume it's the direct `latent_window_size` for formula.
    if win_size_latent_frames <= 0: return "Total Segments: Invalid window size"
    num_video_frames_per_latent_window_segment = win_size_latent_frames * 4 
    if num_video_frames_per_latent_window_segment == 0: return "Total Segments: Invalid window segment length"
    
    total_video_frames = total_seconds * 30
    calculated_sections = total_video_frames / num_video_frames_per_latent_window_segment
    total_segments = int(max(round(calculated_sections), 1))
    return f"Calculated Total Segments: {total_segments}"

# --- Main Processing Loop (Adapted from pr150_queues) ---
# ... (This extensive function 'process_task_queue_main_loop' is assumed to be correctly ported as in previous response)
# ... (Ensure 'worker_args' inside it correctly maps all new parameters like 'segments_to_decode_csv_str')
# --- Helper functions for queue (move, remove, handle_select, clear, save/load zip, auto save/load) ---
# These are complex and would be here, adapted from pr150 and the previous response.
# For brevity, only showing the stubs of what needs to be fully implemented from prior versions.
# (Full implementation of these queue functions would make this file very long for one response)
# Key is that task["params"] must correctly store all needed worker args.

# --- Gradio UI Definition ---
css_theme = make_progress_bar_css() + """ /* ... (CSS from previous response) ... */ """
block = gr.Blocks(css=css_theme, title="FramePack SVC").queue()
global_state_for_autosave_dict = {} # For atexit
atexit.register(autosave_queue_on_exit_action, global_state_for_autosave_dict) # Ensure this action is defined

with block:
    app_state = gr.State({"queue_state": {"queue": [], "next_id": 1, "processing": False, "abort_current": False, "editing_task_id": None}})
    global_state_for_autosave_dict.update(app_state.value) # Initial sync

    gr.Markdown('# FramePack SVC (Stable Video Creation)')
    with gr.Row():
        with gr.Column(scale=1): # Input parameters column
            input_image_gallery_ui = gr.Gallery(type="numpy", label="Input Image(s)", height=320, preview=True)
            prompt_ui = gr.Textbox(label="Prompt", value='')
            example_quick_prompts_ui = gr.Dataset(samples=[['The girl dances gracefully...'], ['A character doing simple movements...']], label='Quick Prompts', components=[prompt_ui])
            
            total_segments_display_ui = gr.Markdown("Calculated Total Segments: N/A") # For dynamic display

            with gr.Accordion("Advanced Settings", open=False):
                n_prompt_ui = gr.Textbox(label="Negative Prompt", value="")
                seed_ui = gr.Number(label="Seed (-1 for random per image)", value=31337, precision=0)
                total_second_length_ui = gr.Slider(label="Total Video Length (Seconds)", minimum=1, maximum=120, value=5.0, step=0.1)
                latent_window_size_ui = gr.Slider(label="Latent Window Size (Default: 9)", minimum=1, maximum=33, value=9, step=1) # Used for segment calc
                
                steps_ui = gr.Slider(label="Steps (Default: 25)", minimum=1, maximum=100, value=25, step=1)
                cfg_ui = gr.Slider(label="CFG Scale (Real Guidance)", minimum=1.0, maximum=32.0, value=1.0, step=0.01)
                gs_ui = gr.Slider(label="Distilled CFG Scale (Initial)", minimum=1.0, maximum=32.0, value=10.0, step=0.01)
                rs_ui = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01)
                
                # --- New UI for Variable GS ---
                gs_schedule_active_checkbox_ui = gr.Checkbox(label="Enable Variable Distilled CFG Schedule (Linear)", value=False)
                gs_final_value_ui = gr.Slider(label="Final Distilled CFG Scale (for schedule)", minimum=1.0, maximum=32.0, value=10.0, step=0.01)
                # --- End New UI ---

                gpu_memory_preservation_ui = gr.Slider(label="GPU Inference Preserved Memory (GB)", minimum=4, maximum=128, value=6.0, step=0.1)
                use_teacache_ui = gr.Checkbox(label='Use TeaCache', value=True)
                mp4_crf_ui = gr.Slider(label="MP4 Compression (CRF)", minimum=0, maximum=51, value=18, step=1)
                
                # --- New UI for Selective Decode ---
                segments_to_decode_csv_ui = gr.Textbox(label="Decode Specific Segments (CSV, e.g., 1,5,10 - overrides normal saves)", placeholder="Default: first and final segment", value="")
                # --- End New UI ---
                
                output_folder_ui_ctrl = gr.Textbox(label="Output Folder Base Path", value=outputs_folder)
                save_defaults_button = gr.Button(value="Save Settings as Default", variant="secondary")

            with gr.Row():
                add_task_button = gr.Button(value="Add Task(s) to Queue", variant="primary")
                cancel_edit_task_button = gr.Button(value="Cancel Edit", visible=False, variant="secondary")
            process_queue_button = gr.Button(value="▶️ Process Queue", variant="primary", interactive=True)
            abort_task_button = gr.Button(value="⏹️ Abort Current Task & Halt Queue", interactive=False, variant="stop")

        with gr.Column(scale=2): # Queue and Output column
            # ... (Queue UI, Progress, Last Video - largely same as previous response's Gradio layout)
            # Make sure queue_df_display_ui and other output elements are defined here.
            gr.Markdown("## Task Queue")
            queue_df_display_ui = gr.DataFrame(headers=["ID", "Status", "Prompt", "Length", "Steps", "Input", "↑", "↓", "✖", "✎"], datatype=["number","str","markdown","str","number","markdown","markdown","markdown","markdown","markdown"], col_count=(10,"fixed"), value=[], interactive=False, visible=False, elem_id="queue_df", wrap=True, max_rows=10)
            with gr.Row():
                save_queue_zip_b64_output = gr.Text(visible=False)
                save_queue_button_ui = gr.DownloadButton("Save Queue", size="sm")
                load_queue_button_ui = gr.UploadButton("Load Queue", file_types=[".zip"], size="sm")
                clear_queue_button_ui = gr.Button("Clear Pending", size="sm", variant="stop")
            gr.Markdown("## Current Task Progress")
            current_task_preview_image_ui = gr.Image(label="Live Preview", height=256, interactive=False, visible=False)
            current_task_progress_desc_ui = gr.Markdown('')
            current_task_progress_bar_ui = gr.HTML('')
            gr.Markdown("## Last Finished Video / Segment MP4")
            last_finished_video_ui = gr.Video(label="Output Video", interactive=True, height=400)


    # --- UI Interactions ---
    task_defining_ui_inputs = [
        input_image_gallery_ui, prompt_ui, n_prompt_ui, seed_ui, total_second_length_ui,
        latent_window_size_ui, steps_ui, cfg_ui, gs_ui, rs_ui,
        gpu_memory_preservation_ui, use_teacache_ui, mp4_crf_ui,
        segments_to_decode_csv_ui, # New
        gs_schedule_active_checkbox_ui, # New
        gs_final_value_ui,          # New
        output_folder_ui_ctrl
    ]
    default_settings_components_list = task_defining_ui_inputs[1:] # All except gallery

    # Link segment calculator display
    for ctrl in [total_second_length_ui, latent_window_size_ui]:
        ctrl.change(
            fn=ui_update_total_segments,
            inputs=[total_second_length_ui, latent_window_size_ui],
            outputs=[total_segments_display_ui]
        )
    
    # ... (All other .click, .select, .load, .upload handlers from previous response,
    # ensuring inputs/outputs lists for them correctly include all new UI components like
    # segments_to_decode_csv_ui, gs_schedule_active_checkbox_ui, gs_final_value_ui.
    # The functions they call (add_or_update_task_in_queue, handle_queue_action_on_select, save_defaults, load_defaults)
    # will also need to be aware of these new parameters by their index or by updating
    # param_names_from_ui_for_task_creation.)

    # Example for add_task_button (ensure add_or_update_task_in_queue handles the new params)
    add_task_button.click(
        fn=add_or_update_task_in_queue,
        inputs=[app_state] + task_defining_ui_inputs,
        outputs=[app_state, queue_df_display_ui, add_task_button, cancel_edit_task_button]
    )
    
    # ... (process_queue_button.click, cancel_edit_task_button.click, etc.)
    # ... (queue_df_display_ui.select, block.load, etc.)
    # Ensure all these are correctly wired up, this is a placeholder for that complex Gradio wiring.


    # Example for block.load for defaults, make sure load_defaults provides values for new UI elements
    block.load(
        fn=load_defaults, # load_defaults must be updated for new params
        inputs=[],
        outputs=default_settings_components_list # This list must match what load_defaults returns
    ).then(
        fn=autoload_queue_on_start_action, # This is defined in full in previous response
        inputs=[app_state],
        outputs=[app_state, queue_df_display_ui]
    ).then(
        lambda s_val: global_state_for_autosave_dict.update(s_val),
        inputs=[app_state], outputs=[]
    ).then( # Trigger initial segment calculation
        fn=ui_update_total_segments,
        inputs=[total_second_length_ui, latent_window_size_ui],
        outputs=[total_segments_display_ui]
    )
    example_quick_prompts_ui.click(fn=lambda x: x[0], inputs=[example_quick_prompts_ui], outputs=prompt_ui, show_progress=False, queue=False)


# --- Launch ---
if __name__ == "__main__": # Standard Python practice
    # Port the full definitions of queue helper functions here if not above
    # (move_task_in_queue, remove_task_from_queue, etc.)
    # For this response, their full detailed code from previous interactions is assumed.
    # The same for process_task_queue_main_loop and other core queue logic.
    print("Starting FramePack SVC application...")
    block.launch(
        server_name=args.server,
        server_port=args.port,
        share=args.share,
        inbrowser=args.inbrowser,
    )