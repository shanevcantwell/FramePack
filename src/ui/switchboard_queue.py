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
        components[K.CLEAR_IMAGE_BUTTON],
        components[K.DOWNLOAD_IMAGE_BUTTON],
        components[K.SAVE_QUEUE_BUTTON],
        components[K.CLEAR_QUEUE_BUTTON],
    ]

    default_keys_map = workspace_manager.get_default_values_map()
    full_workspace_ui_components = [components[key] for key in default_keys_map.keys()]
    task_defining_ui_inputs = [components[K.INPUT_IMAGE_DISPLAY]] + full_workspace_ui_components
    lora_ui_controls = [components[K.LORA_NAME], components[K.LORA_WEIGHT], components[K.LORA_TARGETS]]

    # Define the output lists for various queue actions to ensure consistency.
    # These lists must match the return signatures of the functions in queue.py
    add_task_outputs = (
        [components[k] for k in [K.APP_STATE, K.QUEUE_DF, K.INPUT_IMAGE_DISPLAY, K.IMAGE_FILE_INPUT]] +
        full_workspace_ui_components +
        [components[k] for k in [K.CLEAR_IMAGE_BUTTON, K.DOWNLOAD_IMAGE_BUTTON, K.ADD_TASK_BUTTON, K.CANCEL_EDIT_TASK_BUTTON]]
    )

    process_q_outputs = [
        components[K.APP_STATE], components[K.QUEUE_DF], components[K.LAST_FINISHED_VIDEO],
        components[K.CURRENT_TASK_PREVIEW_IMAGE], components[K.CURRENT_TASK_PROGRESS_DESCRIPTION],
        components[K.CURRENT_TASK_PROGRESS_BAR], components[K.PROCESS_QUEUE_BUTTON], components[K.CREATE_PREVIEW_BUTTON], components[K.CLEAR_QUEUE_BUTTON]
    ]

    (components[K.ADD_TASK_BUTTON].click(
        fn=queue_actions.add_or_update_task_in_queue, inputs=task_defining_ui_inputs, outputs=add_task_outputs
    ).then(
        fn=event_handlers.update_button_states, inputs=[components[K.APP_STATE], components[K.INPUT_IMAGE_DISPLAY], components[K.QUEUE_DF]],
        outputs=button_state_outputs
    ).then(
        fn=event_handlers.ui_update_total_segments,
        inputs=[components[K.VIDEO_LENGTH_SLIDER], components[K.LATENT_WINDOW_SIZE_SLIDER], components[K.FPS_SLIDER]],
        outputs=[components[K.TOTAL_SEGMENTS_DISPLAY]]
    ))

    (components[K.PROCESS_QUEUE_BUTTON].click(
        fn=queue_processing.process_task_queue_and_listen, inputs=lora_ui_controls, outputs=process_q_outputs
    ).then(
        fn=event_handlers.update_button_states, inputs=[components[K.APP_STATE], components[K.INPUT_IMAGE_DISPLAY], components[K.QUEUE_DF]],
        outputs=button_state_outputs
    ))

    (components[K.CREATE_PREVIEW_BUTTON].click(
        fn=queue_processing.request_preview_generation_action, inputs=None, outputs=None
    ).then(
        fn=event_handlers.update_button_states, inputs=[components[K.APP_STATE], components[K.INPUT_IMAGE_DISPLAY], components[K.QUEUE_DF]],
        outputs=button_state_outputs
    ))

    (components[K.CANCEL_EDIT_TASK_BUTTON].click(
        fn=queue_actions.cancel_edit_mode_action, inputs=None, outputs=add_task_outputs
    ).then(
        fn=event_handlers.ui_update_total_segments,
        inputs=[components[K.VIDEO_LENGTH_SLIDER], components[K.LATENT_WINDOW_SIZE_SLIDER], components[K.FPS_SLIDER]],
        outputs=[components[K.TOTAL_SEGMENTS_DISPLAY]]
    ))

    (components[K.CLEAR_QUEUE_BUTTON].click(
        fn=queue_actions.clear_task_queue_action, inputs=None, outputs=[components[K.APP_STATE], components[K.QUEUE_DF]]
    ).then(
        fn=event_handlers.update_button_states, inputs=[components[K.APP_STATE], components[K.INPUT_IMAGE_DISPLAY], components[K.QUEUE_DF]],
        outputs=button_state_outputs
    ))

    (components[K.SAVE_QUEUE_BUTTON].click(
        fn=queue_actions.save_queue_to_zip, inputs=None, outputs=[components[K.APP_STATE], components[K.QUEUE_DOWNLOADER]], show_progress=True
    ).then(
        fn=None, inputs=None, outputs=None,
        js="() => { document.getElementById('queue_downloader_hidden_file').querySelector('a[download]').click(); }"
    ))

    (components[K.LOAD_QUEUE_BUTTON].upload(
        fn=queue_actions.load_queue_from_zip, inputs=[components[K.LOAD_QUEUE_BUTTON]], outputs=[components[K.APP_STATE], components[K.QUEUE_DF]]
    ).then(
        fn=event_handlers.update_button_states, inputs=[components[K.APP_STATE], components[K.INPUT_IMAGE_DISPLAY], components[K.QUEUE_DF]], outputs=button_state_outputs
    ))

    (components[K.QUEUE_DF].select(
        fn=queue_actions.handle_queue_action_on_select, inputs=task_defining_ui_inputs, outputs=add_task_outputs
    ).then(
        fn=event_handlers.update_button_states, inputs=[components[K.APP_STATE], components[K.INPUT_IMAGE_DISPLAY], components[K.QUEUE_DF]],
        outputs=button_state_outputs
    ).then(
        fn=event_handlers.ui_update_total_segments,
        inputs=[components[K.VIDEO_LENGTH_SLIDER], components[K.LATENT_WINDOW_SIZE_SLIDER], components[K.FPS_SLIDER]],
        outputs=[components[K.TOTAL_SEGMENTS_DISPLAY]]
    ))