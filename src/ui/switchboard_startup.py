# ui/switchboard_startup.py
import gradio as gr
import logging

from .enums import ComponentKey as K
from . import workspace as workspace_manager
from . import event_handlers
from . import shared_state as shared_state_module
from . import switchboard_helpers # Import the helper module

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
    # button_state_outputs is now handled within switchboard_helpers.chain_event_updates

    settings_path, image_path = gr.State(), gr.State()
    # Initial load event for settings and image paths
    initial_load_event = block.load(
        fn=workspace_manager.load_workspace_on_start,
        inputs=None,
        outputs=[settings_path, image_path]
    )

    # First .then() to load settings based on the path
    settings_load_event = initial_load_event.then(
        fn=workspace_manager.load_settings_from_file,
        inputs=[settings_path],
        outputs=workspace_ui_outputs
    )

    # Second .then() to load image based on the path
    image_load_event = settings_load_event.then(
        fn=workspace_manager.load_image_from_path,
        inputs=[image_path],
        outputs=image_ui_outputs
    )

    # Now, chain the standard UI updates using the helper, off the event that
    # completes the image loading. This ensures consistent handling.
    switchboard_helpers.chain_event_updates(
        image_load_event,
        components,
        update_segments=True # Ensure segments are updated
    )

    # The final update_button_states and total segments calculations are now handled
    # by the chain_event_updates helper, off the image_load_event.
    # We no longer need the explicit .then() calls for them here.

    shutdown_inputs = [components[K.INPUT_IMAGE_DISPLAY]] + workspace_ui_outputs
    logger.info("Startup events wired.")
