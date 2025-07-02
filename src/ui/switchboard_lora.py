# ui/switchboard_lora.py
import gradio as gr
import logging

from .enums import ComponentKey as K
from . import lora as lora_manager

logger = logging.getLogger(__name__)

def wire_events(components: dict):
    """Wires up the LoRA management UI events."""
    logger.info("Wiring LoRA events...")
    lora_upload_and_refresh_outputs = [
        components[K.APP_STATE],
        components[K.LORA_NAME_STATE],
        components[K.LORA_ROW],
        components[K.LORA_NAME],
        components[K.LORA_WEIGHT],
        components[K.LORA_TARGETS]
    ]
    components[K.LORA_UPLOAD_BUTTON].upload(
        fn=lora_manager.handle_lora_upload_and_update_ui,
        inputs=[components[K.APP_STATE], components[K.LORA_UPLOAD_BUTTON]],
        outputs=lora_upload_and_refresh_outputs
    )