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
            components[K.GS_FINAL_UI]: gr.update(visible=show_final_gs, interactive=show_final_gs),
            components[K.ROLL_OFF_START_UI]: gr.update(visible=show_rolloff_sliders),
            components[K.ROLL_OFF_FACTOR_UI]: gr.update(visible=show_rolloff_sliders),
        }

    components[K.GS_SCHEDULE_SHAPE_UI].change(
        fn=update_scheduler_visibility,
        inputs=[components[K.GS_SCHEDULE_SHAPE_UI]],
        outputs=[components[k] for k in [K.GS_FINAL_UI, K.ROLL_OFF_START_UI, K.ROLL_OFF_FACTOR_UI]]
    )

    segment_recalc_triggers = [components[K.TOTAL_SECOND_LENGTH_UI], components[K.LATENT_WINDOW_SIZE_UI], components[K.FPS_UI]]
    for ctrl in segment_recalc_triggers:
        ctrl.change(fn=event_handlers.ui_update_total_segments, inputs=segment_recalc_triggers, outputs=[components[K.TOTAL_SEGMENTS_DISPLAY_UI]])