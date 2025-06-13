# goan.py
# Main application entry point for the goan video generation UI.
# This script is responsible for initializing models, creating the UI,
# wiring up event handlers, and launching the Gradio application.

# --- Python Standard Library Imports ---
import os
import gradio as gr
import torch
import argparse
import atexit

# --- Local Application Imports ---
# Import managers for different UI sections and shared state
from ui import layout as layout_manager
from ui import metadata as metadata_manager
from ui import queue as queue_manager
from ui import workspace as workspace_manager
from ui import shared_state

# --- Diffusers and Helper Imports ---
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer, SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, DynamicSwapInstaller
from diffusers_helper.gradio.progress_bar import make_progress_bar_css

# --- Environment Setup ---
os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="goan: FramePack-based Video Generation UI")
parser.add_argument('--share', action='store_true', default=False, help="Enable Gradio sharing link.")
parser.add_argument("--server", type=str, default='127.0.0.1', help="Server name to bind to.")
parser.add_argument("--port", type=int, required=False, help="Port to run the server on.")
parser.add_argument("--inbrowser", action='store_true', default=False, help="Launch in browser automatically.")
args = parser.parse_args()
print(f"goan launching with args: {args}")

# --- Model Loading ---
print("Initializing models...")
free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60
print(f'Free VRAM {free_mem_gb} GB, High-VRAM Mode: {high_vram}')
shared_state.models = {
    'text_encoder': LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu(),
    'text_encoder_2': CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu(),
    'tokenizer': LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer'),
    'tokenizer_2': CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2'),
    'vae': AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu(),
    'feature_extractor': SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor'),
    'image_encoder': SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu(),
    'transformer': HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu(),
    'high_vram_flag': high_vram
}
print("Models loaded to CPU. Configuring...")
for model_name in ['vae', 'text_encoder', 'text_encoder_2', 'image_encoder', 'transformer']:
    shared_state.models[model_name].eval()
if not high_vram:
    shared_state.models['vae'].enable_slicing(); shared_state.models['vae'].enable_tiling()
shared_state.models['transformer'].high_quality_fp32_output_for_inference = False
for model_name, dtype in [('transformer', torch.bfloat16), ('vae', torch.float16), ('image_encoder', torch.float16), ('text_encoder', torch.float16), ('text_encoder_2', torch.float16)]:
    shared_state.models[model_name].to(dtype=dtype)
for model in shared_state.models.values():
    if isinstance(model, torch.nn.Module): model.requires_grad_(False)
if not high_vram:
    print("Low VRAM mode: Installing DynamicSwap.")
    DynamicSwapInstaller.install_model(shared_state.models['transformer'], device=gpu)
    DynamicSwapInstaller.install_model(shared_state.models['text_encoder'], device=gpu)
else:
    print("High VRAM mode: Moving all models to GPU.")
    for model_name in ['text_encoder', 'text_encoder_2', 'image_encoder', 'vae', 'transformer']:
        shared_state.models[model_name].to(gpu)
print("Model configuration and placement complete.")


# --- UI Helper Functions ---
def patched_video_is_playable(video_filepath): return True
gr.processing_utils.video_is_playable = patched_video_is_playable

def ui_update_total_segments(total_seconds_ui, latent_window_size_ui):
    """Calculates and formats the total segment count for display in the UI."""
    try:
        total_segments = int(max(round((total_seconds_ui * 30) / (latent_window_size_ui * 4)), 1))
        return f"Calculated Total Segments: {total_segments}"
    except (TypeError, ValueError): return "Segments: Invalid input"


# --- UI Creation and Event Wiring ---
print("Creating UI layout...")
# Create the UI by calling the layout manager. This returns the block and a dictionary of components.
ui_components = layout_manager.create_ui()
block = ui_components['block']

# Define lists of components for easier wiring of events
creative_ui_keys = ['prompt_ui', 'n_prompt_ui', 'total_second_length_ui', 'seed_ui', 'preview_frequency_ui', 'segments_to_decode_csv_ui', 'gs_ui', 'gs_schedule_shape_ui', 'gs_final_ui', 'steps_ui', 'cfg_ui', 'rs_ui']
environment_ui_keys = ['use_teacache_ui', 'use_fp32_transformer_output_checkbox_ui', 'gpu_memory_preservation_ui', 'mp4_crf_ui', 'output_folder_ui_ctrl', 'latent_window_size_ui']
full_workspace_ui_keys = creative_ui_keys + environment_ui_keys

creative_ui_components = [ui_components[key] for key in creative_ui_keys]
full_workspace_ui_components = [ui_components[key] for key in full_workspace_ui_keys]
task_defining_ui_inputs = [ui_components['input_image_gallery_ui']] + full_workspace_ui_components

# Define output lists for complex Gradio calls
process_queue_outputs_list = [ui_components[key] for key in ['app_state', 'queue_df_display_ui', 'last_finished_video_ui', 'current_task_preview_image_ui', 'current_task_progress_desc_ui', 'current_task_progress_bar_ui', 'process_queue_button', 'abort_task_button', 'reset_ui_button']]
queue_df_select_outputs_list = [ui_components[key] for key in ['app_state', 'queue_df_display_ui', 'input_image_gallery_ui'] + full_workspace_ui_keys + ['add_task_button', 'cancel_edit_task_button', 'last_finished_video_ui']]

# Wire up all the UI events to their handler functions in the respective managers
with block:
    # Workspace Manager Events
    ui_components['save_workspace_button'].click(fn=workspace_manager.save_workspace, inputs=full_workspace_ui_components, outputs=None)
    ui_components['load_workspace_button'].click(fn=workspace_manager.load_workspace, inputs=None, outputs=full_workspace_ui_components)
    ui_components['save_as_default_button'].click(fn=workspace_manager.save_as_default_workspace, inputs=full_workspace_ui_components, outputs=None)
    
    # Metadata Manager Events
    ui_components['input_image_gallery_ui'].upload(fn=metadata_manager.handle_image_upload_for_metadata, inputs=[ui_components['input_image_gallery_ui']], outputs=[ui_components['metadata_modal']])
    ui_components['confirm_metadata_btn'].click(fn=metadata_manager.apply_and_hide_modal, inputs=[ui_components['input_image_gallery_ui']], outputs=[ui_components['metadata_modal']] + creative_ui_components)
    ui_components['cancel_metadata_btn'].click(fn=lambda: gr.update(visible=False), inputs=None, outputs=ui_components['metadata_modal'])

    # Queue Manager Events
    ui_components['add_task_button'].click(fn=queue_manager.add_or_update_task_in_queue, inputs=[ui_components['app_state']] + task_defining_ui_inputs, outputs=[ui_components['app_state'], ui_components['queue_df_display_ui'], ui_components['add_task_button'], ui_components['cancel_edit_task_button']])
    ui_components['process_queue_button'].click(fn=queue_manager.process_task_queue_main_loop, inputs=[ui_components['app_state']], outputs=process_queue_outputs_list)
    ui_components['cancel_edit_task_button'].click(fn=queue_manager.cancel_edit_mode_action, inputs=[ui_components['app_state']], outputs=[ui_components['app_state'], ui_components['queue_df_display_ui'], ui_components['add_task_button'], ui_components['cancel_edit_task_button']])
    ui_components['abort_task_button'].click(fn=queue_manager.abort_current_task_processing_action, inputs=[ui_components['app_state']], outputs=[ui_components['app_state'], ui_components['abort_task_button']])
    ui_components['clear_queue_button_ui'].click(fn=queue_manager.clear_task_queue_action, inputs=[ui_components['app_state']], outputs=[ui_components['app_state'], ui_components['queue_df_display_ui']])
    ui_components['save_queue_button_ui'].click(fn=queue_manager.save_queue_to_zip, inputs=[ui_components['app_state']], outputs=[ui_components['app_state'], ui_components['save_queue_zip_b64_output']]).then(fn=None, inputs=[ui_components['save_queue_zip_b64_output']], outputs=None, js="""(b64) => { if(!b64) return; const blob = new Blob([Uint8Array.from(atob(b64), c => c.charCodeAt(0))], {type: 'application/zip'}); const url = URL.createObjectURL(blob); const a = document.createElement('a'); a.href=url; a.download='goan_queue.zip'; a.click(); URL.revokeObjectURL(url); }""")
    ui_components['load_queue_button_ui'].upload(fn=queue_manager.load_queue_from_zip, inputs=[ui_components['app_state'], ui_components['load_queue_button_ui']], outputs=[ui_components['app_state'], ui_components['queue_df_display_ui']])
    ui_components['queue_df_display_ui'].select(fn=queue_manager.handle_queue_action_on_select, inputs=[ui_components['app_state']] + task_defining_ui_inputs, outputs=queue_df_select_outputs_list)

    # Other UI Event Handlers
    ui_components['gs_schedule_shape_ui'].change(fn=lambda choice: gr.update(interactive=(choice != "Off")), inputs=[ui_components['gs_schedule_shape_ui']], outputs=[ui_components['gs_final_ui']])
    for ctrl_key in ['total_second_length_ui', 'latent_window_size_ui']:
        ui_components[ctrl_key].change(fn=ui_update_total_segments, inputs=[ui_components['total_second_length_ui'], ui_components['latent_window_size_ui']], outputs=[ui_components['total_segments_display_ui']])

    refresh_image_path_state = gr.State(None)
    ui_components['reset_ui_button'].click(fn=workspace_manager.save_ui_and_image_for_refresh, inputs=task_defining_ui_inputs, outputs=None).then(fn=None, inputs=None, outputs=None, js="() => { window.location.reload(); }")

    # --- Application Startup and Shutdown ---
    autoload_outputs = [ui_components[k] for k in ['app_state', 'queue_df_display_ui', 'process_queue_button', 'abort_task_button', 'last_finished_video_ui']]
    (block.load(fn=workspace_manager.load_workspace_on_start, inputs=[], outputs=[refresh_image_path_state] + full_workspace_ui_components)
        .then(fn=workspace_manager.load_image_from_path, inputs=[refresh_image_path_state], outputs=[ui_components['input_image_gallery_ui']])
        .then(fn=queue_manager.autoload_queue_on_start_action, inputs=[ui_components['app_state']], outputs=autoload_outputs)
        .then(lambda s_val: shared_state.global_state_for_autosave.update(s_val), inputs=[ui_components['app_state']], outputs=None)
        .then(fn=ui_update_total_segments, inputs=[ui_components['total_second_length_ui'], ui_components['latent_window_size_ui']], outputs=[ui_components['total_segments_display_ui']]))
    
    atexit.register(queue_manager.autosave_queue_on_exit_action, shared_state.global_state_for_autosave)

# --- Application Launch ---
if __name__ == "__main__":
    print("Starting goan FramePack UI...")
    expanded_outputs_folder = os.path.abspath(os.path.expanduser(workspace_manager.outputs_folder))
    block.launch(server_name=args.server, server_port=args.port, share=args.share, inbrowser=args.inbrowser, allowed_paths=[expanded_outputs_folder])
