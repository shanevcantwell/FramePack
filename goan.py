# goan.py (Corrected)
# Main application orchestrator for the goan video generation UI.

# --- Python Standard Library Imports ---
import os
import atexit
import gradio as gr
import tempfile
import threading

# --- Local Application Imports ---
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
os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))
args = args_manager.parse_args()
model_loader.load_and_configure_models()

# --- UI Creation and Component Mapping ---
print("Creating UI layout...")
gr.processing_utils.video_is_playable = lambda video_filepath: True
ui_components = layout_manager.create_ui()
block = ui_components['block']

creative_ui_components = [ui_components[key] for key in shared_state.CREATIVE_UI_KEYS]
full_workspace_ui_components = [ui_components[key] for key in shared_state.ALL_TASK_UI_KEYS]
task_defining_ui_inputs = [ui_components['input_image_display_ui']] + full_workspace_ui_components

process_q_outputs = [
    ui_components['app_state'],
    ui_components['queue_df_display_ui'],
    ui_components['last_finished_video_ui'],
    ui_components['current_task_preview_image_ui'],
    ui_components['current_task_progress_desc_ui'],
    ui_components['current_task_progress_bar_ui'],
    ui_components['process_queue_button'], # Still needs to be in this list for its interactive/variant updates
    ui_components['abort_task_button'],
    ui_components['clear_queue_button_ui']
]
autoload_outputs = [
    ui_components['app_state'], ui_components['queue_df_display_ui'],
    ui_components['process_queue_button'], ui_components['abort_task_button'],
    ui_components['last_finished_video_ui']
]

# Outputs for a full UI refresh without page reload
all_ui_refresh_outputs = (
    [ui_components['input_image_display_ui']] +
    full_workspace_ui_components +
    [ui_components['clear_image_button_ui'], ui_components['download_image_button_ui'],
     ui_components['image_file_input_ui'], ui_components['total_segments_display_ui']]
)


# --- Event Wiring ---
print("Wiring UI events...")
with block:
    # --- Workspace Manager Events ---
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

    ui_components['load_workspace_button'].upload(
        fn=workspace_manager.load_workspace,
        inputs=[ui_components['load_workspace_button']],
        outputs=full_workspace_ui_components
    ).then(
        # NEW: Update button states after loading workspace
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
        ui_components['add_task_button'], # This output will be overwritten by the next .then()
        ui_components['input_image_display_ui'],
        ui_components['extracted_metadata_state']
    ]
    (ui_components['image_file_input_ui'].upload(
        fn=event_handlers.process_upload_and_show_image,
        inputs=[ui_components['image_file_input_ui']],
        outputs=[ui_components['image_file_input_ui'], ui_components['input_image_display_ui'],
                 ui_components['clear_image_button_ui'], ui_components['download_image_button_ui'],
                 ui_components['add_task_button'], # This is temporary, will be corrected by next .then()
                 ui_components['input_image_display_ui'], ui_components['metadata_prompt_preview_ui'],
                 ui_components['extracted_metadata_state'], ui_components['modal_trigger_box']]
    ).then(
        # NEW: Update button states after image upload
        fn=event_handlers.update_button_states,
        inputs=[ui_components['app_state'], ui_components['input_image_display_ui'], ui_components['queue_df_display_ui']],
        outputs=[ui_components['add_task_button'], ui_components['process_queue_button']]
    ))

    (ui_components['clear_image_button_ui'].click(
        fn=event_handlers.clear_image_action,
        inputs=None,
        outputs=clear_button_outputs
    ).then(
        # NEW: Update button states after image clear
        fn=event_handlers.update_button_states,
        inputs=[ui_components['app_state'], ui_components['input_image_display_ui'], ui_components['queue_df_display_ui']],
        outputs=[ui_components['add_task_button'], ui_components['process_queue_button']]
    ))

    (ui_components['download_image_button_ui'].click(fn=event_handlers.prepare_image_for_download, inputs=[ui_components['input_image_display_ui'], gr.State(shared_state.CREATIVE_UI_KEYS)] + creative_ui_components, outputs=[ui_components['image_downloader_ui']]).then(fn=None, inputs=None, outputs=None, js="""() => { const all_buttons = document.querySelectorAll('.gradio-container button'); const download_button = Array.from(all_buttons).find(el => el.innerText === 'Download'); if (download_button) { const new_download_button_parent = download_button.parentElement.parentElement; const link = new_download_button_parent.querySelector('a[download]'); if(link) { link.click(); } } }"""))
    ui_components['modal_trigger_box'].change(fn=lambda x: gr.update(visible=True) if x else gr.update(visible=False), inputs=[ui_components['modal_trigger_box']], outputs=[ui_components['metadata_modal']], api_name=False, queue=False)
    (ui_components['confirm_metadata_btn'].click(fn=metadata_manager.ui_load_params_from_image_metadata, inputs=[ui_components['extracted_metadata_state']], outputs=creative_ui_components).then(fn=lambda: None, inputs=None, outputs=[ui_components['modal_trigger_box']]))
    ui_components['cancel_metadata_btn'].click(fn=lambda: None, inputs=None, outputs=[ui_components['modal_trigger_box']])

    # --- Queue Manager Events ---
    select_q_outputs = ([ui_components[k] for k in ['app_state', 'queue_df_display_ui', 'input_image_display_ui']] + full_workspace_ui_components + [ui_components[k] for k in ['clear_image_button_ui', 'download_image_button_ui', 'add_task_button', 'cancel_edit_task_button']])
    (ui_components['add_task_button'].click(
        fn=queue_manager.add_or_update_task_in_queue,
        inputs=[ui_components['app_state']] + task_defining_ui_inputs,
        outputs=[ui_components['app_state'], ui_components['queue_df_display_ui'], ui_components['add_task_button'], ui_components['cancel_edit_task_button']]
    ).then(
        # NEW: Update button states after adding a task
        fn=event_handlers.update_button_states,
        inputs=[ui_components['app_state'], ui_components['input_image_display_ui'], ui_components['queue_df_display_ui']],
        outputs=[ui_components['add_task_button'], ui_components['process_queue_button']]
    ))

    (ui_components['process_queue_button'].click(
        fn=queue_manager.process_task_queue_main_loop,
        inputs=[ui_components['app_state']],
        outputs=process_q_outputs # This already includes process_queue_button for interactive/variant updates
    ).then(
        # NEW: Ensure button states are correctly set after processing concludes (success/abort/error)
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
        # NEW: Update button states after aborting a task
        fn=event_handlers.update_button_states,
        inputs=[ui_components['app_state'], ui_components['input_image_display_ui'], ui_components['queue_df_display_ui']],
        outputs=[ui_components['add_task_button'], ui_components['process_queue_button']]
    )

    (ui_components['clear_queue_button_ui'].click(
        fn=queue_manager.clear_task_queue_action,
        inputs=[ui_components['app_state']],
        outputs=[ui_components['app_state'], ui_components['queue_df_display_ui']]
    ).then(
        # NEW: Update button states after clearing the queue
        fn=event_handlers.update_button_states,
        inputs=[ui_components['app_state'], ui_components['input_image_display_ui'], ui_components['queue_df_display_ui']],
        outputs=[ui_components['add_task_button'], ui_components['process_queue_button']]
    ))

    ui_components['save_queue_button_ui'].click(fn=queue_manager.save_queue_to_zip, inputs=[ui_components['app_state']], outputs=[ui_components['save_queue_zip_b64_output']]).then(fn=None, inputs=[ui_components['save_queue_zip_b64_output']], outputs=None, js="""(b64) => { if(!b64) return; const blob = new Blob([Uint8Array.from(atob(b64), c => c.charCodeAt(0))], {type: 'application/zip'}); const url = URL.createObjectURL(blob); const a = document.createElement('a'); a.href=url; a.download='goan_queue.zip'; a.click(); URL.revokeObjectURL(url); }""")
    ui_components['load_queue_button_ui'].upload(fn=queue_manager.load_queue_from_zip, inputs=[ui_components['app_state'], ui_components['load_queue_button_ui']], outputs=[ui_components['app_state'], ui_components['queue_df_display_ui']]) # Autoload state updates here, followed by button updates via new chain below
    ui_components['queue_df_display_ui'].select(fn=queue_manager.handle_queue_action_on_select, inputs=[ui_components['app_state']] + task_defining_ui_inputs, outputs=select_q_outputs) # Still needs to update buttons for move/remove/edit actions

    # NEW: Chain update_button_states after queue_df_display_ui changes from selection actions
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

    # 'Save & Refresh UI' now performs a refresh of UI components without reloading the page.
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
        # NEW: Update button states after UI refresh completes
        fn=event_handlers.update_button_states,
        inputs=[ui_components['app_state'], ui_components['input_image_display_ui'], ui_components['queue_df_display_ui']],
        outputs=[ui_components['add_task_button'], ui_components['process_queue_button']]
    )


    # The relaunch button still forces a reload as its primary function
    ui_components['relaunch_button'].click(
        fn=workspace_manager.save_ui_and_image_for_refresh,
        inputs=task_defining_ui_inputs,
        outputs=None
    ).then(
        fn=None,
        inputs=None,
        outputs=None,
        js="() => { setTimeout(() => { window.location.reload(); }, 500); }"
    )
    shutdown_inputs = [ui_components['app_state']] + task_defining_ui_inputs
    ui_components['shutdown_button'].click(fn=event_handlers.safe_shutdown_action, inputs=shutdown_inputs, outputs=None)

    # --- Application Load/Startup Events ---
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
        # NEW: Update button states on initial load after everything else has settled
        .then(
            fn=event_handlers.update_button_states,
            inputs=[ui_components['app_state'], ui_components['input_image_display_ui'], ui_components['queue_df_display_ui']],
            outputs=[ui_components['add_task_button'], ui_components['process_queue_button']]
        )
        .then(fn=queue_manager.process_task_queue_main_loop, inputs=[ui_components['app_state']], outputs=process_q_outputs, js="""(s) => { if (s && s.queue_state && s.queue_state.processing) { console.log('Reconnecting to processing stream...'); return s; } return null; }"""))

    # Register autosave on exit.
    atexit.register(queue_manager.autosave_queue_on_exit_action, shared_state.global_state_for_autosave)

# --- Application Launch ---
if __name__ == "__main__":
    print("Starting goan FramePack UI...")
    # Define a shared state for graceful exit flag, for main app loop to check
    # and a separate flag for immediate exit.
    # shared_state.interrupt_flag is used for aborting current task.
    # We'll use a new event for signaling a full application exit.
    shared_state.app_exit_event = threading.Event() # NEW: Event for signaling application exit

    def console_listener():
        while True:
            try:
                # Use sys.stdin.readline for better control and non-blocking if needed (though input() is simpler for basic commands)
                # For direct 'ZZ' without Enter, or ':q!' without Enter, this is not possible with basic input().
                # We will require Enter after the command.
                command = input("\n> Console command ('q' or 'quit' for safe exit, 'abort' to stop current task, 'ZZ' to save & exit, ':q!' to exit immediately): ").strip() # Removed .lower() for :q!
                
                if command.lower() in ['q', 'quit']:
                    print("Initiating graceful shutdown...")
                    shared_state.interrupt_flag.set() # Signal abort for any running task
                    shared_state.abort_state['level'] = 2 # Force hard abort for quicker exit
                    shared_state.app_exit_event.set() # Signal main app to exit
                    break # Exit the console listener loop
                elif command.lower() == 'abort':
                    print("Sending graceful abort signal to current task...")
                    shared_state.interrupt_flag.set() # Signal abort
                    shared_state.abort_state['level'] = 1 # Graceful abort
                elif command == 'ZZ': # Case-sensitive for direct 'ZZ' if typed
                    print("Console: 'ZZ' detected. Initiating graceful save and exit...")
                    # Trigger autosave on exit immediately
                    queue_manager.autosave_queue_on_exit_action(shared_state.global_state_for_autosave)
                    print("Autosave complete. Signaling application exit.")
                    shared_state.interrupt_flag.set() # Signal abort for any running task
                    shared_state.abort_state['level'] = 2 # Force hard abort for quicker exit
                    shared_state.app_exit_event.set() # Signal main app to exit
                    break # Exit the console listener loop
                elif command == ':q!': # Case-sensitive for ':q!'
                    print("Console: ':q!' detected. Exiting immediately without saving.")
                    # Do NOT trigger autosave. Force terminate.
                    os._exit(0) # This will exit the process immediately. Use with caution.
                else:
                    print("Unknown command. Valid commands: 'q' (quit), 'abort', 'ZZ', ':q!'.")
            except EOFError: # Ctrl+D or end of input stream
                print("\nEOF detected. Exiting console listener.")
                shared_state.interrupt_flag.set()
                shared_state.abort_state['level'] = 2
                shared_state.app_exit_event.set()
                break
            except Exception as e:
                print(f"Console listener error: {e}")
                time.sleep(1) # Prevent busy-looping on error

    # Start the console listener in a daemon thread
    console_thread = threading.Thread(target=console_listener, daemon=True)
    console_thread.start()

    print("----------------------------------------------------------------------------------------------------")
    print("Service Console: Type 'q' or 'quit' to safely exit, 'abort' to stop current task,")
    print("                 'ZZ' to save & exit, ':q!' to exit immediately (no save).")
    print("----------------------------------------------------------------------------------------------------")

    # This part of the code needs to gracefully exit when shared_state.app_exit_event is set.
    # The block.launch() call itself is usually blocking.
    # For a clean exit initiated from console, you'd typically need to terminate the Gradio server
    # via a signal or a specific Gradio API call, which is not directly exposed for this use case.
    # The os._exit(0) for ':q!' is a forceful exit.
    # For 'q'/'quit'/'ZZ', the atexit handler for autosave will still fire if the process ends naturally.
    # However, if the Gradio server keeps running, the 'app_exit_event' needs to be checked by the main Gradio loop.
    # Gradio's `launch()` is blocking, so a clean exit often requires sending a signal to the process itself.
    # The `atexit` will cover the save on normal Python exit.

    # Example of how you might handle a cleaner exit if block.launch() supported a shutdown event:
    # block.launch(..., shutdown_event=shared_state.app_exit_event) # Hypothetical feature

    # For now, `os._exit(0)` is the only truly immediate exit without relying on Gradio's internal shutdown.
    # For 'q'/'quit'/'ZZ', setting the event and relying on atexit is the closest you can get to graceful from console.
    # The user will still need to Ctrl+C after 'q' or 'ZZ' to terminate the main Gradio process,
    # *unless* Gradio's launch loop itself polls the `shared_state.app_exit_event`.
    # Gradio's launch loop does not expose a way to check an external exit event directly.

    # Therefore, for 'q' and 'ZZ' to actually *shut down the Gradio server*, you will still likely need a Ctrl+C after
    # the save/abort message appears, unless you use os._exit(0) for those too (which defeats 'graceful save').
    # This is a limitation of mixing blocking server calls with console input for full application control.



    initial_output_folder_path = workspace_manager.get_initial_output_folder_from_settings()
    expanded_outputs_folder_for_launch = os.path.abspath(initial_output_folder_path)
    final_allowed_paths = [expanded_outputs_folder_for_launch]
    if args.allowed_output_paths:
        custom_paths = [os.path.abspath(os.path.expanduser(p.strip())) for p in args.allowed_output_paths.split(',') if p.strip()]
        final_allowed_paths.extend(custom_paths)
    final_allowed_paths = list(set(final_allowed_paths))

    print(f"Gradio allowed paths: {final_allowed_paths}")
    block.launch(server_name=args.server, server_port=args.port, share=args.share, inbrowser=args.inbrowser, allowed_paths=final_allowed_paths)