# ui/switchboard.py
# This module acts as a central switchboard for wiring all Gradio events.
# It imports the necessary managers and handlers and connects them to the
# UI components, keeping the main goan.py entry point clean.

import gradio as gr

# Import all necessary modules containing event logic
from . import (
    lora as lora_manager,
    workspace as workspace_manager,
    metadata as metadata_manager,
    queue as queue_manager,
    event_handlers,
    shared_state
)

def _wire_lora_events(components: dict):
    """Wires up the LoRA management UI events."""
    # This list now lives encapsulated within the wiring function.
    lora_upload_and_refresh_outputs = [
        components['app_state'],
        components['lora_name_state'],
        components['lora_row_0'],
        components['lora_name_0'],
        components['lora_weight_0'],
        components['lora_targets_0']
    ]

    components['lora_upload_button_ui'].upload(
        fn=lora_manager.handle_lora_upload_and_update_ui,
        inputs=[
            components['app_state'],
            components['lora_upload_button_ui']
        ],
        outputs=lora_upload_and_refresh_outputs
    )

def _wire_workspace_events(components: dict):
    """Wires up the workspace save/load events."""
    full_workspace_ui_components = [components[key] for key in shared_state.ALL_TASK_UI_KEYS]

    (components['save_workspace_button'].click(
        fn=workspace_manager.save_workspace,
        inputs=full_workspace_ui_components,
        outputs=components['workspace_downloader_ui']
    ).then(
        fn=None,
        inputs=[components['workspace_downloader_ui']],
        outputs=None,
        js="""
        (fileComponentValue) => {
            const downloadContainer = document.getElementById('workspace_downloader_hidden_file');
            if (downloadContainer) {
                const downloadLink = downloadContainer.querySelector('a[download]');
                if (downloadLink) { downloadLink.click(); }
            }
        }
        """
    ))

    components['load_workspace_button'].upload(
        fn=workspace_manager.load_workspace,
        inputs=[components['load_workspace_button']],
        outputs=full_workspace_ui_components
    ).then(
        fn=event_handlers.update_button_states,
        inputs=[components['app_state'], components['input_image_display_ui'], components['queue_df_display_ui']],
        outputs=[components['add_task_button'], components['process_queue_button']]
    )

    components['save_as_default_button'].click(
        fn=workspace_manager.save_as_default_workspace,
        inputs=full_workspace_ui_components,
        outputs=[components['relaunch_notification_md'], components['relaunch_button']]
    )

def _wire_image_and_metadata_events(components: dict):
    """Wires up the main image input and metadata modal events."""
    creative_ui_components = [components[key] for key in shared_state.CREATIVE_UI_KEYS]
    clear_button_outputs = [
        components['image_file_input_ui'], components['input_image_display_ui'],
        components['clear_image_button_ui'], components['download_image_button_ui'],
        components['add_task_button'],
        components['input_image_display_ui'],
        components['extracted_metadata_state']
    ]
    
    upload_outputs = [
        components['image_file_input_ui'], components['input_image_display_ui'],
        components['clear_image_button_ui'], components['download_image_button_ui'],
        components['add_task_button'], components['input_image_display_ui'],
        components['metadata_prompt_preview_ui'],
        components['extracted_metadata_state'], components['modal_trigger_box']
    ]

    (components['image_file_input_ui'].upload(
        fn=event_handlers.process_upload_and_show_image,
        inputs=[components['image_file_input_ui']],
        outputs=upload_outputs
    ).then(
        fn=event_handlers.update_button_states,
        inputs=[components['app_state'], components['input_image_display_ui'], components['queue_df_display_ui']],
        outputs=[components['add_task_button'], components['process_queue_button']]
    ))

    (components['clear_image_button_ui'].click(
        fn=event_handlers.clear_image_action,
        inputs=None,
        outputs=clear_button_outputs
    ).then(
        fn=event_handlers.update_button_states,
        inputs=[components['app_state'], components['input_image_display_ui'], components['queue_df_display_ui']],
        outputs=[components['add_task_button'], components['process_queue_button']]
    ))

    components['download_image_button_ui'].click(
        fn=event_handlers.prepare_image_for_download,
        inputs=(
            [components['input_image_display_ui'], components['app_state'], gr.State(shared_state.CREATIVE_UI_KEYS)] +
            creative_ui_components
        ),
        outputs=components['download_image_button_ui'],
        show_progress=True,
        api_name="download_image_with_metadata"
    )

    # Metadata Modal Events
    components['modal_trigger_box'].change(fn=lambda x: gr.update(visible=True) if x else gr.update(visible=False), inputs=[components['modal_trigger_box']], outputs=[components['metadata_modal']], api_name=False, queue=False)
    (components['confirm_metadata_btn'].click(fn=metadata_manager.ui_load_params_from_image_metadata, inputs=[components['extracted_metadata_state']], outputs=creative_ui_components).then(fn=lambda: None, inputs=None, outputs=[components['modal_trigger_box']]))
    components['cancel_metadata_btn'].click(fn=lambda: None, inputs=None, outputs=[components['modal_trigger_box']])

def _wire_queue_events(components: dict):
    """Wires up all queue management events."""
    full_workspace_ui_components = [components[key] for key in shared_state.ALL_TASK_UI_KEYS]
    task_defining_ui_inputs = [components['input_image_display_ui']] + full_workspace_ui_components
    lora_ui_controls = [
        components['lora_name_0'],
        components['lora_weight_0'],
        components['lora_targets_0']
    ]

    select_q_outputs = (
        [components[k] for k in ['app_state', 'queue_df_display_ui', 'input_image_display_ui']] +
        full_workspace_ui_components +
        [components[k] for k in ['clear_image_button_ui', 'download_image_button_ui', 'add_task_button', 'cancel_edit_task_button']]
    )
    process_q_outputs = [
        components['app_state'], components['queue_df_display_ui'], components['last_finished_video_ui'],
        components['current_task_preview_image_ui'], components['current_task_progress_desc_ui'],
        components['current_task_progress_bar_ui'], components['process_queue_button'],
        components['abort_task_button'], components['clear_queue_button_ui']
    ]
    
    # Event Wiring
    (components['add_task_button'].click(
        fn=queue_manager.add_or_update_task_in_queue,
        inputs=[components['app_state']] + task_defining_ui_inputs,
        outputs=[components['app_state'], components['queue_df_display_ui'], components['add_task_button'], components['cancel_edit_task_button']]
    ).then(
        fn=event_handlers.update_button_states,
        inputs=[components['app_state'], components['input_image_display_ui'], components['queue_df_display_ui']],
        outputs=[components['add_task_button'], components['process_queue_button']]
    ))

    (components['process_queue_button'].click(
        fn=queue_manager.process_task_queue_main_loop,
        inputs=[components['app_state']] + lora_ui_controls,
        outputs=process_q_outputs
    ).then(
        fn=event_handlers.update_button_states,
        inputs=[components['app_state'], components['input_image_display_ui'], components['queue_df_display_ui']],
        outputs=[components['add_task_button'], components['process_queue_button']]
    ))

    components['cancel_edit_task_button'].click(fn=queue_manager.cancel_edit_mode_action, inputs=[components['app_state']], outputs=[components['app_state'], components['queue_df_display_ui'], components['add_task_button'], components['cancel_edit_task_button']])
    
    (components['abort_task_button'].click(
        fn=queue_manager.abort_current_task_processing_action,
        inputs=[components['app_state']],
        outputs=[components['app_state'], components['abort_task_button']]
    ).then(
        fn=event_handlers.update_button_states,
        inputs=[components['app_state'], components['input_image_display_ui'], components['queue_df_display_ui']],
        outputs=[components['add_task_button'], components['process_queue_button']]
    ))

    (components['clear_queue_button_ui'].click(
        fn=queue_manager.clear_task_queue_action,
        inputs=[components['app_state']],
        outputs=[components['app_state'], components['queue_df_display_ui']]
    ).then(
        fn=event_handlers.update_button_states,
        inputs=[components['app_state'], components['input_image_display_ui'], components['queue_df_display_ui']],
        outputs=[components['add_task_button'], components['process_queue_button']]
    ))

    components['save_queue_button_ui'].click(
        fn=queue_manager.save_queue_to_zip,
        inputs=[components['app_state']],
        outputs=[components['app_state'], components['save_queue_button_ui']],
        show_progress=True
    )

    components['load_queue_button_ui'].upload(fn=queue_manager.load_queue_from_zip, inputs=[components['app_state'], components['load_queue_button_ui']], outputs=[components['app_state'], components['queue_df_display_ui']])

    (components['queue_df_display_ui'].select(
        fn=queue_manager.handle_queue_action_on_select,
        inputs=[components['app_state']] + task_defining_ui_inputs,
        outputs=select_q_outputs
    ).then(
        fn=event_handlers.update_button_states,
        inputs=[components['app_state'], components['input_image_display_ui'], components['queue_df_display_ui']],
        outputs=[components['add_task_button'], components['process_queue_button']]
    ))

def _wire_misc_control_events(components: dict):
    """Wires up other miscellaneous UI controls."""
    components['gs_schedule_shape_ui'].change(fn=lambda choice: gr.update(interactive=(choice != "Off")), inputs=[components['gs_schedule_shape_ui']], outputs=[components['gs_final_ui']])

    for ctrl_key in ['total_second_length_ui', 'latent_window_size_ui']:
        components[ctrl_key].change(fn=event_handlers.ui_update_total_segments, inputs=[components['total_second_length_ui'], components['latent_window_size_ui']], outputs=[components['total_segments_display_ui']])


def _wire_app_lifecycle_events(components: dict):
    """Wires app lifecycle events like reset, relaunch, and shutdown."""
    full_workspace_ui_components = [components[key] for key in shared_state.ALL_TASK_UI_KEYS]
    task_defining_ui_inputs = [components['input_image_display_ui']] + full_workspace_ui_components

    (components['reset_ui_button'].click(
        fn=workspace_manager.save_ui_and_image_for_refresh,
        inputs=task_defining_ui_inputs,
        outputs=None
    ).then(
        fn=workspace_manager.load_settings_from_file,
        inputs=gr.State(workspace_manager.UNLOAD_SAVE_FILENAME),
        outputs=full_workspace_ui_components
    ).then(
        fn=workspace_manager.load_image_from_path,
        inputs=[components['input_image_display_ui']],
        outputs=[components['input_image_display_ui']]
    ).then(
        fn=lambda img: gr.update(visible=img is not None), inputs=[components['input_image_display_ui']], outputs=[components['clear_image_button_ui']]
    ).then(
        fn=lambda img: gr.update(visible=img is not None), inputs=[components['input_image_display_ui']], outputs=[components['download_image_button_ui']]
    ).then(
        fn=lambda img: gr.update(visible=img is None), inputs=[components['input_image_display_ui']], outputs=[components['image_file_input_ui']]
    ).then(
        fn=event_handlers.ui_update_total_segments,
        inputs=[components['total_second_length_ui'], components['latent_window_size_ui']],
        outputs=[components['total_segments_display_ui']]
    ))
    
    (components['relaunch_button'].click(
        fn=workspace_manager.save_ui_and_image_for_refresh,
        inputs=task_defining_ui_inputs,
        outputs=None
    ).then(
        fn=None, inputs=None, outputs=None,
        js="""() => { setTimeout(() => { window.location.reload(); }, 500); }"""
    ))
    
    shutdown_inputs = [components['app_state']] + task_defining_ui_inputs
    components['shutdown_button'].click(fn=event_handlers.safe_shutdown_action, inputs=shutdown_inputs, outputs=None)


def wire_all_events(components: dict):
    """
    Main function to orchestrate the wiring of all UI events.
    This is the single entry point called from goan.py.
    """
    block = components['block']
    with block:
        print("Wiring UI events from switchboard...")
        _wire_lora_events(components)
        _wire_workspace_events(components)
        _wire_image_and_metadata_events(components)
        _wire_queue_events(components)
        _wire_misc_control_events(components)
        _wire_app_lifecycle_events(components)
        print("All UI events wired.")