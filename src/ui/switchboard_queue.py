# ui/switchboard_queue.py
import gradio as gr
import logging

from .enums import ComponentKey as K
from . import (
    queue as queue_actions,
    queue_processing,
    event_handlers,
    workspace as workspace_manager,
    shared_state as shared_state_module
)

logger = logging.getLogger(__name__)

def wire_events(components: dict):
    """Wires up all queue management events."""
    logger.info("Wiring queue events...")
    button_state_outputs = [
        components[K.ADD_TASK_BUTTON],
        components[K.PROCESS_QUEUE_BUTTON],
        components[K.CREATE_PREVIEW_BUTTON],
        components[K.CLEAR_IMAGE_BUTTON_UI],
        components[K.DOWNLOAD_IMAGE_BUTTON_UI],
        components[K.SAVE_QUEUE_BUTTON_UI],
        components[K.CLEAR_QUEUE_BUTTON_UI],
    ]

    default_keys_map = workspace_manager.get_default_values_map()
    full_workspace_ui_components = [components[K[key.upper()]] for key in default_keys_map.keys()]
    task_defining_ui_inputs = [components[K.INPUT_IMAGE_DISPLAY_UI]] + full_workspace_ui_components
    lora_ui_controls = [components[K.LORA_NAME_0], components[K.LORA_WEIGHT_0], components[K.LORA_TARGETS_0]]

    # Define the output lists for various queue actions to ensure consistency.
    # These lists must match the return signatures of the functions in queue.py
    add_task_outputs = (
        [components[k] for k in [K.APP_STATE, K.QUEUE_DF_DISPLAY_UI, K.INPUT_IMAGE_DISPLAY_UI, K.IMAGE_FILE_INPUT_UI]] +
        full_workspace_ui_components +
        [components[k] for k in [K.CLEAR_IMAGE_BUTTON_UI, K.DOWNLOAD_IMAGE_BUTTON_UI, K.ADD_TASK_BUTTON, K.CANCEL_EDIT_TASK_BUTTON]]
    )

    process_q_outputs = [
        components[K.APP_STATE], components[K.QUEUE_DF_DISPLAY_UI], components[K.LAST_FINISHED_VIDEO_UI],
        components[K.CURRENT_TASK_PREVIEW_IMAGE_UI], components[K.CURRENT_TASK_PROGRESS_DESC_UI],
        components[K.CURRENT_TASK_PROGRESS_BAR_UI], components[K.PROCESS_QUEUE_BUTTON], components[K.CREATE_PREVIEW_BUTTON], components[K.CLEAR_QUEUE_BUTTON_UI]
    ]

    (components[K.ADD_TASK_BUTTON].click(
        fn=queue_actions.add_or_update_task_in_queue, inputs=task_defining_ui_inputs, outputs=add_task_outputs
    ).then(
        fn=event_handlers.update_button_states, inputs=[components[K.APP_STATE], components[K.INPUT_IMAGE_DISPLAY_UI], components[K.QUEUE_DF_DISPLAY_UI]],
        outputs=button_state_outputs
    ).then(
        fn=event_handlers.ui_update_total_segments,
        inputs=[components[K.TOTAL_SECOND_LENGTH_UI], components[K.LATENT_WINDOW_SIZE_UI], components[K.FPS_UI]],
        outputs=[components[K.TOTAL_SEGMENTS_DISPLAY_UI]]
    ))

    (components[K.PROCESS_QUEUE_BUTTON].click(
        fn=queue_processing.process_task_queue_and_listen, inputs=lora_ui_controls, outputs=process_q_outputs
    ).then(
        fn=event_handlers.update_button_states, inputs=[components[K.APP_STATE], components[K.INPUT_IMAGE_DISPLAY_UI], components[K.QUEUE_DF_DISPLAY_UI]],
        outputs=button_state_outputs
    ))

    (components[K.CREATE_PREVIEW_BUTTON].click(
        fn=queue_processing.request_preview_generation_action, inputs=None, outputs=None
    ).then(
        fn=event_handlers.update_button_states, inputs=[components[K.APP_STATE], components[K.INPUT_IMAGE_DISPLAY_UI], components[K.QUEUE_DF_DISPLAY_UI]],
        outputs=button_state_outputs
    ))

    (components[K.CANCEL_EDIT_TASK_BUTTON].click(
        fn=queue_actions.cancel_edit_mode_action, inputs=None, outputs=add_task_outputs
    ).then(
        fn=event_handlers.ui_update_total_segments,
        inputs=[components[K.TOTAL_SECOND_LENGTH_UI], components[K.LATENT_WINDOW_SIZE_UI], components[K.FPS_UI]],
        outputs=[components[K.TOTAL_SEGMENTS_DISPLAY_UI]]
    ))

    (components[K.CLEAR_QUEUE_BUTTON_UI].click(
        fn=queue_actions.clear_task_queue_action, inputs=None, outputs=[components[K.APP_STATE], components[K.QUEUE_DF_DISPLAY_UI]]
    ).then(
        fn=event_handlers.update_button_states, inputs=[components[K.APP_STATE], components[K.INPUT_IMAGE_DISPLAY_UI], components[K.QUEUE_DF_DISPLAY_UI]],
        outputs=button_state_outputs
    ))

    (components[K.SAVE_QUEUE_BUTTON_UI].click(
        fn=queue_actions.save_queue_to_zip, inputs=None, outputs=[components[K.APP_STATE], components[K.QUEUE_DOWNLOADER_UI]], show_progress=True
    ).then(
        fn=None, inputs=None, outputs=None,
        js="() => { document.getElementById('queue_downloader_hidden_file').querySelector('a[download]').click(); }"
    ))

    (components[K.LOAD_QUEUE_BUTTON_UI].upload(
        fn=queue_actions.load_queue_from_zip, inputs=[components[K.LOAD_QUEUE_BUTTON_UI]], outputs=[components[K.APP_STATE], components[K.QUEUE_DF_DISPLAY_UI]]
    ).then(
        fn=event_handlers.update_button_states, inputs=[components[K.APP_STATE], components[K.INPUT_IMAGE_DISPLAY_UI], components[K.QUEUE_DF_DISPLAY_UI]], outputs=button_state_outputs
    ))

    (components[K.QUEUE_DF_DISPLAY_UI].select(
        fn=queue_actions.handle_queue_action_on_select, inputs=task_defining_ui_inputs, outputs=add_task_outputs
    ).then(
        fn=event_handlers.update_button_states, inputs=[components[K.APP_STATE], components[K.INPUT_IMAGE_DISPLAY_UI], components[K.QUEUE_DF_DISPLAY_UI]],
        outputs=button_state_outputs
    ).then(
        fn=event_handlers.ui_update_total_segments,
        inputs=[components[K.TOTAL_SECOND_LENGTH_UI], components[K.LATENT_WINDOW_SIZE_UI], components[K.FPS_UI]],
        outputs=[components[K.TOTAL_SEGMENTS_DISPLAY_UI]]
    ))