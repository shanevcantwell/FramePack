<<<<<<< Updated upstream
# ui/switchboard_startup.py
import gradio as gr
import logging

from .enums import ComponentKey as K
from . import (
    workspace as workspace_manager,
    event_handlers,
    shared_state as shared_state_module
)

logger = logging.getLogger(__name__)

def wire_events(components: dict):
    """Wires events that run on application load and shutdown."""
    logger.info("Wiring app startup and shutdown events...")
    block = components[K.BLOCK]

    workspace_ui_outputs = [components[key] for key in shared_state_module.ALL_TASK_UI_KEYS]
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

    settings_path, image_path = gr.State(), gr.State()
    (block.load(
        fn=workspace_manager.load_workspace_on_start,
        inputs=None,
        outputs=[settings_path, image_path]
    ).then(
        fn=workspace_manager.load_settings_from_file,
        inputs=[settings_path],
        outputs=workspace_ui_outputs
    ).then(
        fn=workspace_manager.load_image_from_path,
        inputs=[image_path],
        outputs=image_ui_outputs
    ).then(
        fn=event_handlers.ui_update_total_segments,
        inputs=[components[K.TOTAL_SECOND_LENGTH_UI], components[K.LATENT_WINDOW_SIZE_UI], components[K.FPS_UI]],
        outputs=[components[K.TOTAL_SEGMENTS_DISPLAY_UI]]
    ).then(
        fn=event_handlers.update_button_states,
        inputs=[components[K.APP_STATE], components[K.INPUT_IMAGE_DISPLAY_UI], components[K.QUEUE_DF_DISPLAY_UI]],
        outputs=button_state_outputs
    ))

    shutdown_inputs = [components[K.INPUT_IMAGE_DISPLAY_UI]] + workspace_ui_outputs
=======
# ui/switchboard_startup.py
import gradio as gr
import logging

from .enums import ComponentKey as K
from . import (
    workspace as workspace_manager,
    event_handlers,
    shared_state as shared_state_module
)

logger = logging.getLogger(__name__)

def wire_events(components: dict):
    """Wires events that run on application load and shutdown."""
    logger.info("Wiring app startup and shutdown events...")
    block = components[K.BLOCK]

    workspace_ui_outputs = [components[key] for key in shared_state_module.ALL_TASK_UI_KEYS]
    image_ui_outputs = [
        components[K.INPUT_IMAGE_DISPLAY],
        components[K.CLEAR_IMAGE_BUTTON],
        components[K.DOWNLOAD_IMAGE_BUTTON],
        components[K.IMAGE_FILE_INPUT]
    ]
    button_state_outputs = [
        components[K.ADD_TASK_BUTTON],
        components[K.PROCESS_QUEUE_BUTTON],
        components[K.CREATE_PREVIEW_BUTTON],
        components[K.CLEAR_IMAGE_BUTTON],
        components[K.DOWNLOAD_IMAGE_BUTTON],
        components[K.SAVE_QUEUE_BUTTON],
        components[K.CLEAR_QUEUE_BUTTON],
    ]

    settings_path, image_path = gr.State(), gr.State()
    (block.load(
        fn=workspace_manager.load_workspace_on_start,
        inputs=None,
        outputs=[settings_path, image_path]
    ).then(
        fn=workspace_manager.load_settings_from_file,
        inputs=[settings_path],
        outputs=workspace_ui_outputs
    ).then(
        fn=workspace_manager.load_image_from_path,
        inputs=[image_path],
        outputs=image_ui_outputs
    ).then(
        fn=event_handlers.ui_update_total_segments,
        inputs=[components[K.TOTAL_SECOND_LENGTH], components[K.LATENT_WINDOW_SIZE], components[K.FPS]],
        outputs=[components[K.TOTAL_SEGMENTS_DISPLAY]]
    ).then(
        fn=event_handlers.update_button_states,
        inputs=[components[K.APP_STATE], components[K.INPUT_IMAGE_DISPLAY], components[K.QUEUE_DF_DISPLAY]],
        outputs=button_state_outputs
    ))

    shutdown_inputs = [components[K.INPUT_IMAGE_DISPLAY]] + workspace_ui_outputs
>>>>>>> Stashed changes
    components[K.SHUTDOWN_BUTTON].click(fn=event_handlers.safe_shutdown_action, inputs=[components[K.APP_STATE]] + shutdown_inputs, outputs=None)