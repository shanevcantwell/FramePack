# ui/switchboard.py
# This module acts as a central switchboard for wiring all Gradio events.

import logging

from .enums import ComponentKey as K
from . import (
    switchboard_lora,
    switchboard_workspace,
    switchboard_image,
    switchboard_queue,
    switchboard_misc,
    switchboard_startup
)

logger = logging.getLogger(__name__)

def wire_all_events(components: dict):
    """
    Main function to orchestrate the wiring of all UI events by delegating
    to specialized switchboard modules.
    """
    block = components[K.BLOCK]
    with block:
        logger.info("Wiring all UI events from the main switchboard...")

        # Delegate wiring to each specialized module
        switchboard_lora.wire_events(components)
        switchboard_workspace.wire_events(components)
        switchboard_image.wire_events(components)
        switchboard_queue.wire_events(components)
        switchboard_misc.wire_events(components)
        switchboard_startup.wire_events(components)

        logger.info("All UI events have been successfully wired.")