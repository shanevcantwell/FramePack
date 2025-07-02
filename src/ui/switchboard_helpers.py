# ui/switchboard_helpers.py
# Contains helper functions to simplify event wiring in switchboard modules.
import logging
from .enums import ComponentKey as K
from . import event_handlers

logger = logging.getLogger(__name__)
