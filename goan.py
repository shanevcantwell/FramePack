# goan.py (Refactored)
# Main application orchestrator for the goan video generation UI.

# --- Python Standard Library Imports ---
import os
import atexit
import gradio as gr

# --- Local Application Imports ---
# These imports are now clean and directly reflect the abstracted modules.
from core import args as args_manager
from core import model_loader
from ui import (
    layout as layout_manager,
    metadata as metadata_manager,
    queue as queue_manager,
    workspace as workspace_manager,
    event_handlers,
    shared_state
)

# --- Environment Setup & Model Loading ---
# The responsibility of setting HF_HOME is still here, as it's an application-level environment concern.
os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

# Argument parsing is now a simple call to the abstracted module.
args = args_manager.parse_args()

# Model loading and configuration is also a simple call to its dedicated module.
model_loader.load_and_configure_models()

# --- UI Creation and Component Mapping ---
print("Creating UI layout...")
# Monkey-patch for video display in Gradio - this is a Gradio-specific adaptation.
gr.processing_utils.video_is_playable = lambda video_filepath: True
ui_components = layout_manager.create_ui()
block = ui_components['block']

# Create lists of components for easier wiring
creative_ui_components = [ui_components[key] for key in shared_state.CREATIVE_UI_KEYS]
full_workspace_ui_components = [ui_components[key] for key in shared_state.ALL_TASK_UI_KEYS]
task_defining_ui_inputs = [ui_components['input_image_display_ui']] + full_workspace_ui_components

# --- Event Wiring ---
print("Wiring UI events...")
with block:
    # --- Workspace Manager Events ---
    # These calls clearly delegate to the workspace_manager.
    ui_components['save_workspace_button'].click(fn=workspace_manager.save_workspace, inputs=full_workspace_ui_components, outputs=None)
    ui_components['load_workspace_button'].click(fn=workspace_manager.load_workspace, inputs=None, outputs=full_workspace_ui_components)
    ui_components['save_as_default_button'].click(fn=workspace_manager.save_as_default_workspace, inputs=full_workspace_ui_components, outputs=None)

    # --- Image Component & Metadata Modal Events ---
    clear_button_outputs = [
        ui_components['image_file_input_ui'], ui_components['input_image_display_ui'],
        ui_components['clear_image_button_ui'], ui_components['download_image_button_ui'],
        ui_components['add_task_button'], ui_components['input_image_display_ui'],
        ui_components['extracted_metadata_state']
    ]
    # Event handlers are delegated to event_handlers module.
    ui_components['image_file_input_ui'].upload(fn=event_handlers.process_upload_and_show_image, inputs=[ui_components['image_file_input_ui']], outputs=[ui_components['image_file_input_ui'], ui_components['input_image_display_ui'], ui_components['clear_image_button_ui'], ui_components['download_image_button_ui'], ui_components['add_task_button'], ui_components['input_image_display_ui'], ui_components['metadata_prompt_preview_ui'], ui_components['extracted_metadata_state'], ui_components['modal_trigger_box']])
    ui_components['clear_image_button_ui'].click(fn=event_handlers.clear_image_action, inputs=None, outputs=clear_button_outputs)
    # The download button's JS logic remains here as it's deeply tied to Gradio's frontend interaction.
    (ui_components['download_image_button_ui'].click(fn=event_handlers.prepare_image_for_download, inputs=[ui_components['input_image_display_ui'], gr.State(shared_state.CREATIVE_UI_KEYS)] + creative_ui_components, outputs=[ui_components['image_downloader_ui']]).then(fn=None, inputs=None, outputs=None, js="""() => { const all_buttons = document.querySelectorAll('.gradio-container button'); const download_button = Array.from(all_buttons).find(el => el.innerText === 'Download'); if (download_button) { const new_download_button_parent = download_button.parentElement.parentElement; const link = new_download_button_parent.querySelector('a[download]'); if(link) { link.click(); } } }"""))
    ui_components['modal_trigger_box'].change(fn=lambda x: gr.update(visible=True) if x else gr.update(visible=False), inputs=[ui_components['modal_trigger_box']], outputs=[ui_components['metadata_modal']], api_name=False, queue=False)
    # Metadata loading is delegated to metadata_manager.
    (ui_components['confirm_metadata_btn'].click(fn=metadata_manager.ui_load_params_from_image_metadata, inputs=[ui_components['extracted_metadata_state']], outputs=creative_ui_components).then(fn=lambda: None, inputs=None, outputs=[ui_components['modal_trigger_box']]))
    ui_components['cancel_metadata_btn'].click(fn=lambda: None, inputs=None, outputs=[ui_components['modal_trigger_box']])

    # --- Queue Manager Events ---
    # All queue-related actions are delegated to queue_manager.
    process_q_outputs = [
        ui_components['app_state'],
        ui_components['queue_df_display_ui'],
        ui_components['last_finished_video_ui'],
        ui_components['current_task_preview_image_ui'],
        ui_components['current_task_progress_desc_ui'],
        ui_components['current_task_progress_bar_ui'],
        ui_components['process_queue_button'],
        ui_components['abort_task_button'],
        ui_components['clear_queue_button_ui'] # This was the 9th value controlling this button's interactivity
    ]
    select_q_outputs = ([ui_components[k] for k in ['app_state', 'queue_df_display_ui', 'input_image_display_ui']] + full_workspace_ui_components + [ui_components[k] for k in ['clear_image_button_ui', 'download_image_button_ui', 'add_task_button', 'cancel_edit_task_button']])
    ui_components['add_task_button'].click(fn=queue_manager.add_or_update_task_in_queue, inputs=[ui_components['app_state']] + task_defining_ui_inputs, outputs=[ui_components['app_state'], ui_components['queue_df_display_ui'], ui_components['add_task_button'], ui_components['cancel_edit_task_button']])
    # CORRECTED: Renamed process_queue_action to process_task_queue_main_loop
    ui_components['process_queue_button'].click(fn=queue_manager.process_task_queue_main_loop, inputs=[ui_components['app_state']], outputs=process_q_outputs)
    ui_components['cancel_edit_task_button'].click(fn=queue_manager.cancel_edit_mode_action, inputs=[ui_components['app_state']], outputs=[ui_components['app_state'], ui_components['queue_df_display_ui'], ui_components['add_task_button'], ui_components['cancel_edit_task_button']])
    # CORRECTED: Renamed abort_processing_action to abort_current_task_processing_action
    ui_components['abort_task_button'].click(fn=queue_manager.abort_current_task_processing_action, inputs=[ui_components['app_state']], outputs=None)
    ui_components['clear_queue_button_ui'].click(fn=queue_manager.clear_task_queue_action, inputs=[ui_components['app_state']], outputs=[ui_components['app_state'], ui_components['queue_df_display_ui']])
    ui_components['save_queue_button_ui'].click(fn=queue_manager.save_queue_to_zip, inputs=[ui_components['app_state']], outputs=[ui_components['save_queue_zip_b64_output']]).then(fn=None, inputs=[ui_components['save_queue_zip_b64_output']], outputs=None, js="""(b64) => { if(!b64) return; const blob = new Blob([Uint8Array.from(atob(b64), c => c.charCodeAt(0))], {type: 'application/zip'}); const url = URL.createObjectURL(blob); const a = document.createElement('a'); a.href=url; a.download='goan_queue.zip'; a.click(); URL.revokeObjectURL(url); }""")
    ui_components['load_queue_button_ui'].upload(fn=queue_manager.load_queue_from_zip, inputs=[ui_components['app_state'], ui_components['load_queue_button_ui']], outputs=[ui_components['app_state'], ui_components['queue_df_display_ui']])
    ui_components['queue_df_display_ui'].select(fn=queue_manager.handle_queue_action_on_select, inputs=[ui_components['app_state']] + task_defining_ui_inputs, outputs=select_q_outputs)
    
    # --- Other UI Control Events ---
    ui_components['gs_schedule_shape_ui'].change(fn=lambda choice: gr.update(interactive=(choice != "Off")), inputs=[ui_components['gs_schedule_shape_ui']], outputs=[ui_components['gs_final_ui']])
    for ctrl_key in ['total_second_length_ui', 'latent_window_size_ui']:
        ui_components[ctrl_key].change(fn=event_handlers.ui_update_total_segments, inputs=[ui_components['total_second_length_ui'], ui_components['latent_window_size_ui']], outputs=[ui_components['total_segments_display_ui']])
    ui_components['reset_ui_button'].click(fn=workspace_manager.save_ui_and_image_for_refresh, inputs=task_defining_ui_inputs, outputs=None).then(fn=None, inputs=None, outputs=None, js="() => { setTimeout(() => { window.location.reload(); }, 500); }")

    # --- Application Load/Startup Events ---
    refresh_image_path_state = gr.State(None)
    autoload_outputs = [
        ui_components['app_state'], ui_components['queue_df_display_ui'],
        ui_components['process_queue_button'], ui_components['abort_task_button'],
        ui_components['last_finished_video_ui']
]    # Startup logic, using abstracted modules for workspace and queue management.
    (block.load(fn=workspace_manager.load_workspace_on_start, inputs=[], outputs=[refresh_image_path_state] + full_workspace_ui_components)
        .then(fn=workspace_manager.load_image_from_path, inputs=[refresh_image_path_state], outputs=[ui_components['input_image_display_ui']])
        .then(fn=lambda img: gr.update(visible=img is not None), inputs=[ui_components['input_image_display_ui']], outputs=[ui_components['clear_image_button_ui']])
        .then(fn=lambda img: gr.update(visible=img is not None), inputs=[ui_components['input_image_display_ui']], outputs=[ui_components['download_image_button_ui']])
        .then(fn=lambda img: gr.update(visible=img is None), inputs=[ui_components['input_image_display_ui']], outputs=[ui_components['image_file_input_ui']])
        # CORRECTED: Renamed load_queue_on_startup to autoload_queue_on_start_action
        .then(fn=queue_manager.autoload_queue_on_start_action, inputs=[ui_components['app_state']], outputs=autoload_outputs)
        .then(lambda s_val: shared_state.global_state_for_autosave.update(s_val), inputs=[ui_components['app_state']], outputs=None)
        .then(fn=event_handlers.ui_update_total_segments, inputs=[ui_components['total_second_length_ui'], ui_components['latent_window_size_ui']], outputs=[ui_components['total_segments_display_ui']])
        # CORRECTED: Renamed process_queue_action to process_task_queue_main_loop
        .then(fn=queue_manager.process_task_queue_main_loop, inputs=[ui_components['app_state']], outputs=process_q_outputs, js="""(s) => { if (s && s.queue_state && s.queue_state.processing) { console.log('Reconnecting to processing stream...'); return s; } return null; }"""))
    
    # Register autosave on exit.
    atexit.register(queue_manager.autosave_queue_on_exit_action, shared_state.global_state_for_autosave)

# --- Application Launch ---
if __name__ == "__main__":
    print("Starting goan FramePack UI...")
    # These path operations are related to Gradio's launch parameters, which are
    # specific to the main application entry point.
    initial_output_folder_path = workspace_manager.get_initial_output_folder_from_settings()
    expanded_outputs_folder_for_launch = os.path.abspath(initial_output_folder_path)
    final_allowed_paths = [expanded_outputs_folder_for_launch]
    if args.allowed_output_paths:
        custom_paths = [os.path.abspath(os.path.expanduser(p.strip())) for p in args.allowed_output_paths.split(',') if p.strip()]
        final_allowed_paths.extend(custom_paths)
    final_allowed_paths = list(set(final_allowed_paths))
    
    print(f"Gradio allowed paths: {final_allowed_paths}")
    # The actual Gradio launch call.
    block.launch(server_name=args.server, server_port=args.port, share=args.share, inbrowser=args.inbrowser, allowed_paths=final_allowed_paths)