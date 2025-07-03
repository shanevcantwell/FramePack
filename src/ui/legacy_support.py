# ui/legacy_support.py
# Contains helper functions for backward compatibility.
import logging

logger = logging.getLogger(__name__)

def convert_legacy_params(params: dict):
    """
    Checks for and converts legacy parameter formats to their modern equivalents
    by cycling through a dictionary of known legacy keys. This function
    modifies the params dictionary in-place.
    """
    # --- 1. Handle simple key renames ---
    # This map defines old keys and the new keys they should be renamed to.
    key_rename_map = {
        # 'negative_prompt': 'n_prompt',
        # Add other simple renames here in the future, e.g., 'old_name': 'new_name'
    }

    for old_key, new_key in key_rename_map.items():
        if old_key in params:
            # Move the value from the old key to the new key and remove the old one.
            params[new_key] = params.pop(old_key)
            logger.info(f"Mapped legacy parameter '{old_key}' to '{new_key}'.")

    # --- 2. Handle more complex value conversions ---
    # This logic handles the case where not just the key changed, but the value type did too.
    legacy_schedule_key = 'gs_schedule_active'
    new_schedule_key = 'gs_schedule_shape_ui'

    value_to_check = None
    if legacy_schedule_key in params:
        value_to_check = params.pop(legacy_schedule_key)
    # Also check if the modern key exists but with an old boolean value
    elif new_schedule_key in params and isinstance(params[new_schedule_key], bool):
        value_to_check = params[new_schedule_key]

    if isinstance(value_to_check, bool):
        # Convert the legacy boolean to the new string value
        params[new_schedule_key] = "Linear" if value_to_check else "Off"
        logger.info(f"Converted legacy boolean schedule value '{value_to_check}' to '{params[new_schedule_key]}'.")

def convert_legacy_worker_params(params: dict):
    """
    Checks for and converts legacy parameter formats that use worker keys.
    This is typically used for parameters loaded from image metadata.
    This function modifies the params dictionary in-place.

    Args:
        params (dict): A dictionary of parameters with worker keys (e.g., 'gs_schedule_shape').
    """
    # 1. Handle simple key renames for worker parameters.
    key_rename_map = {
        'negative_prompt': 'n_prompt',
    }
    for old_key, new_key in key_rename_map.items():
        if old_key in params:
            params[new_key] = params.pop(old_key)
            logger.info(f"Mapped legacy worker parameter '{old_key}' to '{new_key}'.")

    worker_key = 'gs_schedule_shape'
    if worker_key in params and isinstance(params[worker_key], bool):
        legacy_bool_val = params[worker_key]
        params[worker_key] = "Linear" if legacy_bool_val else "Off"
        logger.info(f"Converted legacy boolean worker param '{worker_key}' from '{legacy_bool_val}' to '{params[worker_key]}'.")