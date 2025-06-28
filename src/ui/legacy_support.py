# ui/legacy_support.py
# Contains helper functions for backward compatibility.
import logging

logger = logging.getLogger(__name__)

def convert_legacy_params(params: dict):
    """
    Checks for and converts legacy parameter formats to their modern equivalents.
    This function modifies the params dictionary in-place.

    Args:
        params (dict): A dictionary of parameters loaded from metadata or a workspace.
                       The keys are expected to be UI component keys (e.g., 'gs_schedule_shape_ui').
    """
    # The old parameter name was 'gs_schedule_active' and its value was boolean.
    # The new name is 'gs_schedule_shape_ui' and its value is a string.
    legacy_key = 'gs_schedule_active'
    new_key = 'gs_schedule_shape_ui'

    value_to_check = None
    if legacy_key in params:
        value_to_check = params.pop(legacy_key) # Remove old key
    elif new_key in params and isinstance(params[new_key], bool):
        value_to_check = params[new_key]

    if isinstance(value_to_check, bool):
        params[new_key] = "Linear" if value_to_check else "Off"
        logger.info(f"Converted legacy boolean schedule '{value_to_check}' to '{params[new_key]}'.")

def convert_legacy_worker_params(params: dict):
    """
    Checks for and converts legacy parameter formats that use worker keys.
    This is typically used for parameters loaded from image metadata.
    This function modifies the params dictionary in-place.

    Args:
        params (dict): A dictionary of parameters with worker keys (e.g., 'gs_schedule_shape').
    """
    worker_key = 'gs_schedule_shape'
    if worker_key in params and isinstance(params[worker_key], bool):
        legacy_bool_val = params[worker_key]
        params[worker_key] = "Linear" if legacy_bool_val else "Off"
        logger.info(f"Converted legacy boolean worker param '{worker_key}' from '{legacy_bool_val}' to '{params[worker_key]}'.")