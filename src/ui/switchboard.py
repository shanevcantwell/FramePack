# ui/switchboard.py
# This module acts as a central switchboard for wiring all Gradio events.

import gradio as gr

from .enums import ComponentKey as K
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
    lora_upload_and_refresh_outputs = [
        components[K.APP_STATE],
        components[K.LORA_NAME_STATE],
        components[K.LORA_ROW_0],
        components[K.LORA_NAME_0],
        components[K.LORA_WEIGHT_0],
        components[K.LORA_TARGETS_0]
    ]
    components[K.LORA_UPLOAD_BUTTON_UI].upload(
        fn=lora_manager.handle_lora_upload_and_update_ui,
        inputs=[components[K.APP_STATE], components[K.LORA_UPLOAD_BUTTON_UI]],
        outputs=lora_upload_and_refresh_outputs
    )

def _wire_workspace_events(components: dict):
    """Wires up the workspace save/load events."""
    button_state_outputs = [
        components[K.ADD_TASK_BUTTON],
        components[K.PROCESS_QUEUE_BUTTON],
        components[K.CREATE_PREVIEW_BUTTON],
        components[K.CLEAR_IMAGE_BUTTON_UI],
        components[K.DOWNLOAD_IMAGE_BUTTON_UI],
        components[K.SAVE_QUEUE_BUTTON_UI],
        components[K.CLEAR_QUEUE_BUTTON_UI],
    ]
    full_workspace_ui_components = [components[key] for key in shared_state.ALL_TASK_UI_KEYS]
    (components[K.SAVE_WORKSPACE_BUTTON].click(
        fn=workspace_manager.save_workspace,
        inputs=full_workspace_ui_components,
        outputs=components[K.WORKSPACE_DOWNLOADER_UI]
    ).then(
        fn=None, inputs=[components[K.WORKSPACE_DOWNLOADER_UI]], outputs=None,
        js="(file) => { document.getElementById('workspace_downloader_hidden_file').querySelector('a[download]').click(); }"
    ))
    (components[K.LOAD_WORKSPACE_BUTTON].upload(
        fn=workspace_manager.load_workspace,
        inputs=[components[K.LOAD_WORKSPACE_BUTTON]],
        outputs=full_workspace_ui_components
    ).then(
        fn=event_handlers.update_button_states,
        inputs=[components[K.APP_STATE], components[K.INPUT_IMAGE_DISPLAY_UI], components[K.QUEUE_DF_DISPLAY_UI]],
        outputs=button_state_outputs
    ))
    components[K.SAVE_AS_DEFAULT_BUTTON].click(
        fn=workspace_manager.save_as_default_workspace,
        inputs=full_workspace_ui_components,
        outputs=[components[K.RELAUNCH_NOTIFICATION_MD], components[K.RELAUNCH_BUTTON]]
    )

def _wire_image_and_metadata_events(components: dict):
    """Wires up the main image input and metadata modal events."""
    button_state_outputs = [
        components[K.ADD_TASK_BUTTON],
        components[K.PROCESS_QUEUE_BUTTON],
        components[K.CREATE_PREVIEW_BUTTON],
        components[K.CLEAR_IMAGE_BUTTON_UI],
        components[K.DOWNLOAD_IMAGE_BUTTON_UI],
        components[K.SAVE_QUEUE_BUTTON_UI],
        components[K.CLEAR_QUEUE_BUTTON_UI],
    ]
    creative_ui_components = [components[key] for key in shared_state.CREATIVE_UI_KEYS]
    
    clear_button_outputs = [
        components[K.IMAGE_FILE_INPUT_UI],
        components[K.INPUT_IMAGE_DISPLAY_UI],
        components[K.CLEAR_IMAGE_BUTTON_UI],
        components[K.DOWNLOAD_IMAGE_BUTTON_UI],
        components[K.ADD_TASK_BUTTON],
        components[K.EXTRACTED_METADATA_STATE]
    ]

    upload_outputs = [
        components[K.IMAGE_FILE_INPUT_UI],
        components[K.INPUT_IMAGE_DISPLAY_UI],
        components[K.CLEAR_IMAGE_BUTTON_UI],
        components[K.DOWNLOAD_IMAGE_BUTTON_UI],
        components[K.ADD_TASK_BUTTON],
        components[K.METADATA_PROMPT_PREVIEW_UI],
        components[K.EXTRACTED_METADATA_STATE],
        components[K.MODAL_TRIGGER_BOX]
    ]

    (components[K.IMAGE_FILE_INPUT_UI].upload(
        fn=event_handlers.process_upload_and_show_image,
        inputs=[components[K.IMAGE_FILE_INPUT_UI]],
        outputs=upload_outputs
    ).then(
        fn=event_handlers.update_button_states,
        inputs=[components[K.APP_STATE], components[K.INPUT_IMAGE_DISPLAY_UI], components[K.QUEUE_DF_DISPLAY_UI]],
        outputs=button_state_outputs
    ))

    (components[K.CLEAR_IMAGE_BUTTON_UI].click(
        fn=event_handlers.clear_image_action, inputs=None, outputs=clear_button_outputs
    ).then(
        fn=event_handlers.update_button_states,
        inputs=[components[K.APP_STATE], components[K.INPUT_IMAGE_DISPLAY_UI], components[K.QUEUE_DF_DISPLAY_UI]],
        outputs=button_state_outputs
    ))

    (components[K.DOWNLOAD_IMAGE_BUTTON_UI].click(
        fn=event_handlers.prepare_image_for_download,
        inputs=([components[K.INPUT_IMAGE_DISPLAY_UI], components[K.APP_STATE], gr.State(shared_state.CREATIVE_UI_KEYS)] + creative_ui_components),
        outputs=components[K.IMAGE_DOWNLOADER_UI], show_progress=True, api_name="download_image_with_metadata"
    ).then(
        fn=None, inputs=None, outputs=None,
        js="() => { document.getElementById('image_downloader_hidden_file').querySelector('a[download]').click(); }"
    ))

    components[K.MODAL_TRIGGER_BOX].change(fn=lambda x: gr.update(visible=True) if x else gr.update(visible=False), inputs=[components[K.MODAL_TRIGGER_BOX]], outputs=[components[K.METADATA_MODAL]], api_name=False, queue=False)
    (components[K.CONFIRM_METADATA_BTN].click(fn=metadata_manager.ui_load_params_from_image_metadata, inputs=[components[K.EXTRACTED_METADATA_STATE]], outputs=creative_ui_components).then(fn=lambda: None, inputs=None, outputs=[components[K.MODAL_TRIGGER_BOX]]))
    components[K.CANCEL_METADATA_BTN].click(fn=lambda: None, inputs=None, outputs=[components[K.MODAL_TRIGGER_BOX]])

def _wire_queue_events(components: dict):
    """Wires up all queue management events."""
    button_state_outputs = [
        components[K.ADD_TASK_BUTTON],
        components[K.PROCESS_QUEUE_BUTTON],
        components[K.CREATE_PREVIEW_BUTTON],
        components[K.CLEAR_IMAGE_BUTTON_UI],
        components[K.DOWNLOAD_IMAGE_BUTTON_UI],
        components[K.SAVE_QUEUE_BUTTON_UI],
        components[K.CLEAR_QUEUE_BUTTON_UI],
        components[K.QUEUE_DOWNLOADER_UI] # Added for one-click download
    ]
    full_workspace_ui_components = [components[key] for key in shared_state.ALL_TASK_UI_KEYS]
    task_defining_ui_inputs = [components[K.INPUT_IMAGE_DISPLAY_UI]] + full_workspace_ui_components
    lora_ui_controls = [components[K.LORA_NAME_0], components[K.LORA_WEIGHT_0], components[K.LORA_TARGETS_0]]

    select_q_outputs = (
        [components[k] for k in [K.APP_STATE, K.QUEUE_DF_DISPLAY_UI, K.INPUT_IMAGE_DISPLAY_UI]] +
        full_workspace_ui_components +
        [components[k] for k in [K.CLEAR_IMAGE_BUTTON_UI, K.DOWNLOAD_IMAGE_BUTTON_UI, K.ADD_TASK_BUTTON, K.CANCEL_EDIT_TASK_BUTTON]]
    )
    process_q_outputs = [
        components[K.APP_STATE], components[K.QUEUE_DF_DISPLAY_UI], components[K.LAST_FINISHED_VIDEO_UI],
        components[K.CURRENT_TASK_PREVIEW_IMAGE_UI], components[K.CURRENT_TASK_PROGRESS_DESC_UI],
        components[K.CURRENT_TASK_PROGRESS_BAR_UI], components[K.PROCESS_QUEUE_BUTTON], components[K.CREATE_PREVIEW_BUTTON], components[K.CLEAR_QUEUE_BUTTON_UI]
    ]
    (components[K.ADD_TASK_BUTTON].click(
        fn=queue_manager.add_or_update_task_in_queue, inputs=[components[K.APP_STATE]] + task_defining_ui_inputs,
        outputs=[components[K.APP_STATE], components[K.QUEUE_DF_DISPLAY_UI], components[K.ADD_TASK_BUTTON], components[K.CANCEL_EDIT_TASK_BUTTON]]
    ).then(
        fn=event_handlers.update_button_states, inputs=[components[K.APP_STATE], components[K.INPUT_IMAGE_DISPLAY_UI], components[K.QUEUE_DF_DISPLAY_UI]],
        outputs=button_state_outputs
    ))
    (components[K.PROCESS_QUEUE_BUTTON].click(
        fn=queue_manager.process_task_queue_main_loop, inputs=[components[K.APP_STATE]] + lora_ui_controls, outputs=process_q_outputs
    ).then(
        fn=event_handlers.update_button_states, inputs=[components[K.APP_STATE], components[K.INPUT_IMAGE_DISPLAY_UI], components[K.QUEUE_DF_DISPLAY_UI]],
        outputs=button_state_outputs
    ))
    (components[K.CREATE_PREVIEW_BUTTON].click(
        fn=queue_manager.request_preview_generation_action, inputs=[components[K.APP_STATE]], outputs=[components[K.APP_STATE]]
    ).then(
        fn=event_handlers.update_button_states, inputs=[components[K.APP_STATE], components[K.INPUT_IMAGE_DISPLAY_UI], components[K.QUEUE_DF_DISPLAY_UI]],
        outputs=button_state_outputs
    ))
    (components[K.CLEAR_QUEUE_BUTTON_UI].click(
        fn=queue_manager.clear_task_queue_action, inputs=[components[K.APP_STATE]], outputs=[components[K.APP_STATE], components[K.QUEUE_DF_DISPLAY_UI]]
    ).then(
        fn=event_handlers.update_button_states, inputs=[components[K.APP_STATE], components[K.INPUT_IMAGE_DISPLAY_UI], components[K.QUEUE_DF_DISPLAY_UI]],
        outputs=button_state_outputs
    ))
    # Re-wired for one-click download. The button click prepares the file and outputs it
    # to a hidden component, which is then "clicked" by the JS to trigger the download.
    (components[K.SAVE_QUEUE_BUTTON_UI].click(
        fn=queue_manager.save_queue_to_zip,
        inputs=[components[K.APP_STATE]],
        outputs=[components[K.APP_STATE], components[K.QUEUE_DOWNLOADER_UI]],
        show_progress=True
    ).then(
        fn=None, inputs=None, outputs=None,
        js="() => { document.getElementById('queue_downloader_hidden_file').querySelector('a[download]').click(); }"
    ))
    (components[K.LOAD_QUEUE_BUTTON_UI].upload(
        fn=queue_manager.load_queue_from_zip, inputs=[components[K.APP_STATE], components[K.LOAD_QUEUE_BUTTON_UI]], outputs=[components[K.APP_STATE], components[K.QUEUE_DF_DISPLAY_UI]]
    ).then(
        fn=event_handlers.update_button_states, inputs=[components[K.APP_STATE], components[K.INPUT_IMAGE_DISPLAY_UI], components[K.QUEUE_DF_DISPLAY_UI]], outputs=button_state_outputs
    ))
    (components[K.QUEUE_DF_DISPLAY_UI].select(
        fn=queue_manager.handle_queue_action_on_select, inputs=[components[K.APP_STATE]] + task_defining_ui_inputs, outputs=select_q_outputs
    ).then(
        fn=event_handlers.update_button_states, inputs=[components[K.APP_STATE], components[K.INPUT_IMAGE_DISPLAY_UI], components[K.QUEUE_DF_DISPLAY_UI]],
        outputs=button_state_outputs
    ))

def _wire_misc_control_events(components: dict):
    """Wires up other miscellaneous UI controls."""

    def update_scheduler_visibility(choice: str):
        """Shows/hides scheduler sliders based on the selected schedule type."""
        is_linear = (choice == "Linear")
        is_rolloff = (choice == "Roll-off")
        show_final_gs = is_linear or is_rolloff
        show_rolloff_sliders = is_rolloff

        return {
            components[K.GS_FINAL_UI]: gr.update(visible=show_final_gs, interactive=show_final_gs),
            components[K.ROLL_OFF_START_UI]: gr.update(visible=show_rolloff_sliders),
            components[K.ROLL_OFF_FACTOR_UI]: gr.update(visible=show_rolloff_sliders),
        }

    components[K.GS_SCHEDULE_SHAPE_UI].change(
        fn=update_scheduler_visibility,
        inputs=[components[K.GS_SCHEDULE_SHAPE_UI]],
        outputs=[components[k] for k in [K.GS_FINAL_UI, K.ROLL_OFF_START_UI, K.ROLL_OFF_FACTOR_UI]]
    )
    # The controls that trigger a recalculation of total segments and frames.
    segment_recalc_triggers = [
        components[K.TOTAL_SECOND_LENGTH_UI],
        components[K.LATENT_WINDOW_SIZE_UI],
        components[K.FPS_UI]
    ]
    for ctrl in segment_recalc_triggers:
        ctrl.change(
            fn=event_handlers.ui_update_total_segments,
            inputs=segment_recalc_triggers,
            outputs=[components[K.TOTAL_SEGMENTS_DISPLAY_UI]])

def _wire_app_startup_events(components: dict):
    """Wires events that run on application load."""
    block = components[K.BLOCK]
    
    # Define the outputs for each step in the startup chain
    workspace_ui_outputs = [components[key] for key in shared_state.ALL_TASK_UI_KEYS]
    image_ui_outputs = [
        components[K.INPUT_IMAGE_DISPLAY_UI],
        components[K.CLEAR_IMAGE_BUTTON_UI],
        components[K.DOWNLOAD_IMAGE_BUTTON_UI],
        components[K.IMAGE_FILE_INPUT_UI]
    ]
    button_state_outputs = [
        components[K.ADD_TASK_BUTTON],
        components[K.PROCESS_QUEUE_BUTTON],
        components[K.CREATE_PREVIEW_BUTTON],
        components[K.CLEAR_IMAGE_BUTTON_UI],
        components[K.DOWNLOAD_IMAGE_BUTTON_UI],
        components[K.SAVE_QUEUE_BUTTON_UI],
        components[K.CLEAR_QUEUE_BUTTON_UI],
    ]

    # The .load event now uses a more robust, sequential chain of .then() calls.
    # This was reverted from a single consolidated function to fix a Pydantic validation
    # error that occurred when the UI state was not updated predictably.
    # Step 1: Find the paths for the workspace and any refresh image.
    settings_path, image_path = gr.State(), gr.State()
    (block.load(
        fn=workspace_manager.load_workspace_on_start,
        inputs=None,
        outputs=[settings_path, image_path]
    # Step 2: Load the workspace settings from the found path.
    ).then(
        fn=workspace_manager.load_settings_from_file,
        inputs=[settings_path],
        outputs=workspace_ui_outputs
    # Step 3: Load the refresh image from its path.
    ).then(
        fn=workspace_manager.load_image_from_path,
        inputs=[image_path],
        outputs=image_ui_outputs
    # Step 4: Update the calculated segments display based on loaded settings.
    ).then(
        fn=event_handlers.ui_update_total_segments,
        inputs=[components[K.TOTAL_SECOND_LENGTH_UI], components[K.LATENT_WINDOW_SIZE_UI], components[K.FPS_UI]],
        outputs=[components[K.TOTAL_SEGMENTS_DISPLAY_UI]]
    # Step 5: Update all button states based on the loaded workspace and image.
    # This is the crucial step that prevents the "flash and collapse" of the UI.
    ).then(
        fn=event_handlers.update_button_states,
        inputs=[components[K.APP_STATE], components[K.INPUT_IMAGE_DISPLAY_UI], components[K.QUEUE_DF_DISPLAY_UI]],
        outputs=button_state_outputs
    ))

def wire_all_events(components: dict):
    """Main function to orchestrate the wiring of all UI events."""
    block = components[K.BLOCK]
    with block:
        print("Wiring UI events from switchboard...")
        _wire_lora_events(components)
        _wire_workspace_events(components)
        _wire_image_and_metadata_events(components)
        _wire_queue_events(components)
        _wire_misc_control_events(components)
        _wire_app_startup_events(components)
        print("All UI events wired.")