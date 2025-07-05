# ui/switchboard_helpers.py
# Contains helper functions to simplify event wiring in switchboard modules.
import logging
from .enums import ComponentKey as K
from . import event_handlers

logger = logging.getLogger(__name__)

def chain_event_updates(event, components: dict, update_segments: bool = False):
    """
    Chains standard UI updates (buttons, segments) to a Gradio event.
    This helps to keep the switchboard modules DRY.

    Args:
        event: The Gradio event object to chain to (e.g., the result of a .click()).
        components (dict): The dictionary of all UI components.
        update_segments (bool): If True, chains the segment count update as well.
    """
    button_state_outputs = event_handlers.get_button_state_outputs(components)
    event.then(
        fn=event_handlers.update_button_states,
        inputs=[components[K.APP_STATE], components[K.INPUT_IMAGE_DISPLAY], components[K.IMAGE_FILE_INPUT]
                ],
        outputs=button_state_outputs
    )
    if update_segments:
        segment_recalc_inputs = [
            components[K.VIDEO_LENGTH_SLIDER],
            components[K.LATENT_WINDOW_SIZE_SLIDER],
            components[K.FPS_SLIDER]
        ]
        # --- START OF MODIFICATION ---
        # Apply the same lambda pattern here for ui_update_total_segments.
        # It expects 3 arguments, so the lambda needs to consume 4 (implicit + 3 explicit).
        event.then(
            fn=lambda event_output_payload, video_length, latent_window_size, fps: \
                event_handlers.ui_update_total_segments(video_length, latent_window_size, fps),
            inputs=[
                event, # Implicit outputs of 'event' are the first argument to the lambda
                components[K.VIDEO_LENGTH_SLIDER],
                components[K.LATENT_WINDOW_SIZE_SLIDER],
                components[K.FPS_SLIDER]
            ],
            outputs=[components[K.TOTAL_SEGMENTS_DISPLAY]]
        )
        # --- END OF MODIFICATION ---