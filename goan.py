# goan.py
# Main application orchestrator for the goan video generation UI.

# --- Python Standard Library Imports ---
import os
import atexit
import gradio as gr
import tempfile

# --- Local Application Imports ---
from core import args as args_manager
from core import model_loader
from ui import (
    layout as layout_manager,
    metadata as metadata_manager,
    queue as queue_manager,
    workspace as workspace_manager,
    lora as lora_manager,
    event_handlers,
    shared_state
)

# --- Environment Setup & Model Loading ---
os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))
args = args_manager.parse_args()
model_loader.load_and_configure_models()

# --- UI Creation and Component Mapping ---
print("Creating UI layout...")
# This is a workaround to ensure Gradio considers all video files playable.
gr.processing_utils.video_is_playable = lambda video_filepath: True
ui_components = layout_manager.create_ui()
block = ui_components['block']

# Pre-compile a list of all UI controls related to the static LoRA slots.
lora_ui_controls = []
for i in range(5):
    lora_ui_controls.extend([
        ui_components[f'lora_row_{i}'],
        ui_components[f'lora_name_{i}'],
        ui_components[f'lora_weight_{i}'],
        ui_components[f'lora_targets_{i}']
    ])

# Define lists of UI components for consistent use in event handlers.
creative_ui_components = [ui_components[key] for key in shared_state.CREATIVE_UI_KEYS]
full_workspace_ui_components = [ui_components[key] for key in shared_state.ALL_TASK_UI_KEYS]
task_defining_ui_inputs = [ui_components['input_image_display_ui']] + full_workspace_ui_components

# Define lists of outputs for complex queue-related events.
select_q_outputs = (
    [ui_components[k] for k in ['app_state', 'queue_df_display_ui', 'input_image_display_ui']] +
    full_workspace_ui_components +
    [ui_components[k] for k in ['clear_image_button_ui', 'download_image_button_ui', 'add_task_button', 'cancel_edit_task_button']]
)
process_q_outputs = [
    ui_components['app_state'],
    ui_components['queue_df_display_ui'],
    ui_components['last_finished_video_ui'],
    ui_components['current_task_preview_image_ui'],
    ui_components['current_task_progress_desc_ui'],
    ui_components['current_task_progress_bar_ui'],
    ui_components['process_queue_button'],
    ui_components['abort_task_button'],
    ui_components['clear_queue_button_ui']
]
autoload_outputs = [
    ui_components['app_state'],
    ui_components['queue_df_display_ui'],
    ui_components['process_queue_button'],
    ui_components['abort_task_button'],
    ui_components['last_finished_video_ui']
]

# --- Event Wiring ---
print("Wiring UI events...")
with block:

    # --- LoRA Manager Events ---
    # Populates the static UI slots when a LoRA is uploaded.
    ui_components['lora_upload_button_ui'].upload(
        fn=lora_manager.handle_lora_upload,
        inputs=[
            ui_components['app_state'],
            ui_components['lora_upload_button_ui']
        ],
        outputs=[ui_components['app_state'], ui_components['lora_name_state']] + lora_ui_controls
    )

    # --- Workspace Manager Events ---
    # The 'Save Workspace' button saves settings to a temporary file and then
    # uses a JavaScript snippet to trigger the download of that file.
    (ui_components['save_workspace_button'].click(
        fn=workspace_manager.save_workspace,
        inputs=full_workspace_ui_components,
        outputs=ui_components['workspace_downloader_ui']
    ).then(
        fn=None,
        inputs=[ui_components['workspace_downloader_ui']],
        outputs=None,
        js="""
        (fileComponentValue) => {
            const downloadContainer = document.getElementById('workspace_downloader_hidden_file');
            if (downloadContainer) {
                const downloadLink = downloadContainer.querySelector('a[download]');
                if (downloadLink) {
                    downloadLink.click();
                } else {
                    console.error("Download link element not found within #workspace_downloader_hidden_file");
                }
            } else {
                console.error("Hidden download container #workspace_downloader_hidden_file not found.");
            }
        }
        """
    ))

    # 'Load Workspace' uses the UploadButton to load settings from a JSON file.
    ui_components['load_workspace_button'].upload(
        fn=workspace_manager.load_workspace,
        inputs=[ui_components['load_workspace_button']],
        outputs=full_workspace_ui_components
    ).then(
        fn=event_handlers.update_button_states,
        inputs=[ui_components['app_state'], ui_components['input_image_display_ui'], ui_components['queue_df_display_ui']],
        outputs=[ui_components['add_task_button'], ui_components['process_queue_button']]
    )

    ui_components['save_as_default_button'].click(
        fn=workspace_manager.save_as_default_workspace,
        inputs=full_workspace_ui_components,
        outputs=[ui_components['relaunch_notification_md'], ui_components['relaunch_button']]
    )

    # --- Image Component & Metadata Modal Events ---
    clear_button_outputs = [
        ui_components['image_file_input_ui'], ui_components['input_image_display_ui'],
        ui_components['clear_image_button_ui'], ui_components['download_image_button_ui'],
        ui_components['add_task_button'],
        ui_components['input_image_display_ui'],
        ui_components['extracted_metadata_state']
    ]
    (ui_components['image_file_input_ui'].upload(
        fn=event_handlers.process_upload_and_show_image,
        inputs=[ui_components['image_file_input_ui']],
        outputs=[ui_components['image_file_input_ui'], ui_components['input_image_display_ui'],
                 ui_components['clear_image_button_ui'], ui_components['download_image_button_ui'],
                 ui_components['add_task_button'], ui_components['input_image_display_ui'],
                 ui_components['metadata_prompt_preview_ui'],
                 ui_components['extracted_metadata_state'], ui_components['modal_trigger_box']]
    ).then(
        fn=event_handlers.update_button_states,
        inputs=[ui_components['app_state'], ui_components['input_image_display_ui'], ui_components['queue_df_display_ui']],
        outputs=[ui_components['add_task_button'], ui_components['process_queue_button']]
    ))

    (ui_components['clear_image_button_ui'].click(
        fn=event_handlers.clear_image_action,
        inputs=None,
        outputs=clear_button_outputs
    ).then(
        fn=event_handlers.update_button_states,
        inputs=[ui_components['app_state'], ui_components['input_image_display_ui'], ui_components['queue_df_display_ui']],
        outputs=[ui_components['add_task_button'], ui_components['process_queue_button']]
    ))

    # The DownloadButton for the image calls a function to prepare the image with
    # metadata and returns the filepath, which the button component then handles.
    ui_components['download_image_button_ui'].click(
        fn=event_handlers.prepare_image_for_download,
        inputs=(
            [ui_components['input_image_display_ui'], ui_components['app_state'], gr.State(shared_state.CREATIVE_UI_KEYS)] +
            creative_ui_components
        ),
        outputs=ui_components['download_image_button_ui'],
        show_progress=True,
        api_name="download_image_with_metadata"
    )

    # Logic for the metadata confirmation modal.
    ui_components['modal_trigger_box'].change(fn=lambda x: gr.update(visible=True) if x else gr.update(visible=False), inputs=[ui_components['modal_trigger_box']], outputs=[ui_components['metadata_modal']], api_name=False, queue=False)
    (ui_components['confirm_metadata_btn'].click(fn=metadata_manager.ui_load_params_from_image_metadata, inputs=[ui_components['extracted_metadata_state']], outputs=creative_ui_components).then(fn=lambda: None, inputs=None, outputs=[ui_components['modal_trigger_box']]))
    ui_components['cancel_metadata_btn'].click(fn=lambda: None, inputs=None, outputs=[ui_components['modal_trigger_box']])

    # --- Queue Manager Events ---
    # FIX: Wired up the 'Add to Queue' button, which was previously unresponsive.
    (ui_components['add_task_button'].click(
        fn=queue_manager.add_or_update_task_in_queue,
        inputs=[ui_components['app_state']] + task_defining_ui_inputs,
        outputs=[ui_components['app_state'], ui_components['queue_df_display_ui'], ui_components['add_task_button'], ui_components['cancel_edit_task_button']]
    ).then(
        fn=event_handlers.update_button_states,
        inputs=[ui_components['app_state'], ui_components['input_image_display_ui'], ui_components['queue_df_display_ui']],
        outputs=[ui_components['add_task_button'], ui_components['process_queue_button']]
    ))

    # The 'Process Queue' button gathers LoRA settings directly from the UI controls.
    (ui_components['process_queue_button'].click(
        fn=queue_manager.process_task_queue_main_loop,
        inputs=[ui_components['app_state']] + lora_ui_controls,
        outputs=process_q_outputs
    ).then(
        fn=event_handlers.update_button_states,
        inputs=[ui_components['app_state'], ui_components['input_image_display_ui'], ui_components['queue_df_display_ui']],
        outputs=[ui_components['add_task_button'], ui_components['process_queue_button']]
    ))

    ui_components['cancel_edit_task_button'].click(fn=queue_manager.cancel_edit_mode_action, inputs=[ui_components['app_state']], outputs=[ui_components['app_state'], ui_components['queue_df_display_ui'], ui_components['add_task_button'], ui_components['cancel_edit_task_button']])
    ui_components['abort_task_button'].click(
        fn=queue_manager.abort_current_task_processing_action,
        inputs=[ui_components['app_state']],
        outputs=[ui_components['app_state'], ui_components['abort_task_button']]
    ).then(
        fn=event_handlers.update_button_states,
        inputs=[ui_components['app_state'], ui_components['input_image_display_ui'], ui_components['queue_df_display_ui']],
        outputs=[ui_components['add_task_button'], ui_components['process_queue_button']]
    )

    (ui_components['clear_queue_button_ui'].click(
        fn=queue_manager.clear_task_queue_action,
        inputs=[ui_components['app_state']],
        outputs=[ui_components['app_state'], ui_components['queue_df_display_ui']]
    ).then(
        fn=event_handlers.update_button_states,
        inputs=[ui_components['app_state'], ui_components['input_image_display_ui'], ui_components['queue_df_display_ui']],
        outputs=[ui_components['add_task_button'], ui_components['process_queue_button']]
    ))

    ui_components['save_queue_button_ui'].click(
        fn=queue_manager.save_queue_to_zip,
        inputs=[ui_components['app_state']],
        outputs=[ui_components['app_state'], ui_components['save_queue_button_ui']],
        show_progress=True
    )

    ui_components['load_queue_button_ui'].upload(fn=queue_manager.load_queue_from_zip, inputs=[ui_components['app_state'], ui_components['load_queue_button_ui']], outputs=[ui_components['app_state'], ui_components['queue_df_display_ui']])

    # Handles clicks on the small action icons (up, down, delete, edit) within the queue table.
    (ui_components['queue_df_display_ui'].select(
        fn=queue_manager.handle_queue_action_on_select,
        inputs=[ui_components['app_state']] + task_defining_ui_inputs,
        outputs=select_q_outputs
    ).then(
        fn=event_handlers.update_button_states,
        inputs=[ui_components['app_state'], ui_components['input_image_display_ui'], ui_components['queue_df_display_ui']],
        outputs=[ui_components['add_task_button'], ui_components['process_queue_button']]
    ))

    # --- Other UI Control Events ---
    ui_components['gs_schedule_shape_ui'].change(fn=lambda choice: gr.update(interactive=(choice != "Off")), inputs=[ui_components['gs_schedule_shape_ui']], outputs=[ui_components['gs_final_ui']])

    for ctrl_key in ['total_second_length_ui', 'latent_window_size_ui']:
        ui_components[ctrl_key].change(fn=event_handlers.ui_update_total_segments, inputs=[ui_components['total_second_length_ui'], ui_components['latent_window_size_ui']], outputs=[ui_components['total_segments_display_ui']])

    # The 'Reset UI' button saves current settings, reloads them from the saved file,
    # reloads the image, and then updates various UI component states.
    ui_components['reset_ui_button'].click(
        fn=workspace_manager.save_ui_and_image_for_refresh,
        inputs=task_defining_ui_inputs,
        outputs=None
    ).then(
        fn=workspace_manager.load_settings_from_file,
        inputs=gr.State(workspace_manager.UNLOAD_SAVE_FILENAME),
        outputs=full_workspace_ui_components
    ).then(
        fn=workspace_manager.load_image_from_path,
        inputs=[ui_components['input_image_display_ui']],
        outputs=[ui_components['input_image_display_ui']]
    ).then(
        fn=lambda img: gr.update(visible=img is not None), inputs=[ui_components['input_image_display_ui']], outputs=[ui_components['clear_image_button_ui']]
    ).then(
        fn=lambda img: gr.update(visible=img is not None), inputs=[ui_components['input_image_display_ui']], outputs=[ui_components['download_image_button_ui']]
    ).then(
        fn=lambda img: gr.update(visible=img is None), inputs=[ui_components['input_image_display_ui']], outputs=[ui_components['image_file_input_ui']]
    ).then(
        fn=event_handlers.ui_update_total_segments,
        inputs=[ui_components['total_second_length_ui'], ui_components['latent_window_size_ui']],
        outputs=[ui_components['total_segments_display_ui']]
    ).then(
        fn=event_handlers.update_button_states,
        inputs=[ui_components['app_state'], ui_components['input_image_display_ui'], ui_components['queue_df_display_ui']],
        outputs=[ui_components['add_task_button'], ui_components['process_queue_button']]
    )

    # Saves current state and triggers a page reload via JavaScript.
    ui_components['relaunch_button'].click(
        fn=workspace_manager.save_ui_and_image_for_refresh,
        inputs=task_defining_ui_inputs,
        outputs=None
    ).then(
        fn=None,
        inputs=None,
        outputs=None,
        js="""() => { setTimeout(() => { window.location.reload(); }, 500); }"""
    )
    shutdown_inputs = [ui_components['app_state']] + task_defining_ui_inputs
    ui_components['shutdown_button'].click(fn=event_handlers.safe_shutdown_action, inputs=shutdown_inputs, outputs=None)

    # --- Application Load/Startup Events ---
    # This long chain of events runs once when the Gradio application starts.
    refresh_image_path_state = gr.State(None)
    (block.load(fn=workspace_manager.load_workspace_on_start, inputs=[], outputs=[refresh_image_path_state] + full_workspace_ui_components)
        .then(fn=workspace_manager.load_image_from_path, inputs=[refresh_image_path_state], outputs=[ui_components['input_image_display_ui']])
        .then(fn=lambda: gr.update(visible=not shared_state.system_info.get('is_legacy_gpu', False)), inputs=None, outputs=[ui_components['use_fp32_transformer_output_checkbox_ui']])
        .then(fn=lambda img: gr.update(visible=img is not None), inputs=[ui_components['input_image_display_ui']], outputs=[ui_components['clear_image_button_ui']])
        .then(fn=lambda img: gr.update(visible=img is not None), inputs=[ui_components['input_image_display_ui']], outputs=[ui_components['download_image_button_ui']])
        .then(fn=lambda img: gr.update(visible=img is None), inputs=[ui_components['input_image_display_ui']], outputs=[ui_components['image_file_input_ui']])
        .then(fn=queue_manager.autoload_queue_on_start_action, inputs=[ui_components['app_state']], outputs=autoload_outputs)
        .then(lambda s_val: shared_state.global_state_for_autosave.update(s_val), inputs=[ui_components['app_state']], outputs=None)
        .then(fn=event_handlers.ui_update_total_segments, inputs=[ui_components['total_second_length_ui'], ui_components['latent_window_size_ui']], outputs=[ui_components['total_segments_display_ui']])
        .then(
            fn=event_handlers.update_button_states,
            inputs=[ui_components['app_state'], ui_components['input_image_display_ui'], ui_components['queue_df_display_ui']],
            outputs=[ui_components['add_task_button'], ui_components['process_queue_button']]
        )
        # This JS snippet attempts to reconnect to a running generation stream if the page is reloaded.
        .then(fn=queue_manager.process_task_queue_main_loop, inputs=[ui_components['app_state']], outputs=process_q_outputs, js="""(s) => { if (s && s.queue_state && s.queue_state.processing) { console.log('Reconnecting to processing stream...'); return s; } return null; }"""))

    # Register the queue autosave function to be called when the application exits.
    atexit.register(queue_manager.autosave_queue_on_exit_action, shared_state.global_state_for_autosave)

# --- Application Launch ---
if __name__ == "__main__":
    print("Starting goan FramePack UI...")

    initial_output_folder_path = workspace_manager.get_initial_output_folder_from_settings()
    expanded_outputs_folder_for_launch = os.path.abspath(initial_output_folder_path)

    # Compile the list of paths that Gradio is allowed to serve files from.
    final_allowed_paths = [expanded_outputs_folder_for_launch]
    if args.allowed_output_paths:
        custom_paths = [os.path.abspath(os.path.expanduser(p.strip())) for p in args.allowed_output_paths.split(',') if p.strip()]
        final_allowed_paths.extend(custom_paths)

    # Ensure the 'loras' directory is always an allowed path.
    final_allowed_paths.append(lora_manager.LORA_DIR)
    final_allowed_paths = list(set(final_allowed_paths))

    print(f"Gradio allowed paths: {final_allowed_paths}")
    block.launch(server_name=args.server, server_port=args.port, share=args.share, inbrowser=args.inbrowser, allowed_paths=final_allowed_paths)