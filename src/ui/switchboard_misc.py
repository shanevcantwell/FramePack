# ui/switchboard_misc.py
import gradio as gr
import logging

from .enums import ComponentKey as K
from . import event_handlers

logger = logging.getLogger(__name__)

def wire_events(components: dict):
    """Wires up other miscellaneous UI controls."""
    logger.info("Wiring miscellaneous control events...")

    def update_scheduler_visibility(choice: str):
        """Shows/hides scheduler sliders based on the selected schedule type."""
        is_linear = (choice == "Linear")
        is_rolloff = (choice == "Roll-off")
        show_final_gs = is_linear or is_rolloff
        show_rolloff_sliders = is_rolloff

        return {
            components[K.DISTILLED_CFG_END_SLIDER]: gr.update(visible=show_final_gs, interactive=show_final_gs),
            components[K.ROLL_OFF_START_SLIDER]: gr.update(visible=show_rolloff_sliders),
            components[K.ROLL_OFF_FACTOR_SLIDER]: gr.update(visible=show_rolloff_sliders),
        }

        # Random Seed Button: On click, output the number -1 to the SEED component.
    components[K.RANDOM_SEED_BUTTON].click(
        fn=lambda: -1, 
        inputs=None, 
        outputs=[components[K.SEED]]
    )

    # Reuse Seed Button: On click, run the handler function.
    # Input is the state holding the last seed, output is the SEED component.
    components[K.REUSE_SEED_BUTTON].click(
        fn=event_handlers.reuse_last_seed_action,
        inputs=[components[K.LAST_COMPLETED_SEED_STATE]],
        outputs=[components[K.SEED]]
    )

    components[K.VARIABLE_CFG_SHAPE_RADIO].change(
        fn=update_scheduler_visibility,
        inputs=[components[K.VARIABLE_CFG_SHAPE_RADIO]],
        outputs=[components[k] for k in [K.DISTILLED_CFG_END_SLIDER, K.ROLL_OFF_START_SLIDER, K.ROLL_OFF_FACTOR_SLIDER]]
    )

    segment_recalc_triggers = [components[K.VIDEO_LENGTH_SLIDER], components[K.LATENT_WINDOW_SIZE_SLIDER], components[K.FPS_SLIDER]]
    for ctrl in segment_recalc_triggers:
        ctrl.change(fn=event_handlers.ui_update_total_segments, inputs=segment_recalc_triggers, outputs=[components[K.TOTAL_SEGMENTS_DISPLAY]])
