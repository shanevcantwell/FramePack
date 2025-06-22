# ui/legacy_support.py
# NEW FILE: Contains helper functions for backward compatibility.

def convert_legacy_params(params: dict):
    """
    Checks for and converts legacy parameter formats to their modern equivalents.
    This function modifies the params dictionary in-place.

    Args:
        params (dict): A dictionary of parameters loaded from metadata or a workspace.
                       The keys are expected to be UI component keys (e.g., 'gs_schedule_shape_ui').
    """
    # VESTIGIAL: Handle legacy boolean for gs_schedule_shape.
    # The old parameter name was 'gs_schedule_active' and its value was boolean.
    # The new name is 'gs_schedule_shape_ui' and its value is a string.
    legacy_key = 'gs_schedule_active' # This key is from very old workspaces
    new_key = 'gs_schedule_shape_ui'

    value_to_check = None
    if legacy_key in params:
        value_to_check = params.pop(legacy_key) # Remove old key
    elif new_key in params and isinstance(params[new_key], bool):
        value_to_check = params[new_key]

    if isinstance(value_to_check, bool):
        params[new_key] = "Linear" if value_to_check else "Off"
        print(f"INFO: Converted legacy boolean schedule '{value_to_check}' to '{params[new_key]}'.")