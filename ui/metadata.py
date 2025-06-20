# ui/metadata.py
import json
import gradio as gr
from PIL import Image
from PIL.PngImagePlugin import PngInfo

# Import shared state for access to parameter key lists
from . import shared_state

# --- Core Metadata Functions ---

def extract_metadata_from_pil_image(pil_image: Image.Image) -> dict:
    """Extracts a 'parameters' dictionary from a PIL image's text chunk."""
    if pil_image is None:
        return {}

    # The 'text' attribute of a PIL image holds the PNG tEXt chunks.
    pnginfo_data = getattr(pil_image, 'text', None)
    if not isinstance(pnginfo_data, dict):
        return {}

    params_json_str = pnginfo_data.get('parameters')
    if not params_json_str:
        return {}

    try:
        extracted_params = json.loads(params_json_str)
        return extracted_params if isinstance(extracted_params, dict) else {}
    except json.JSONDecodeError as e:
        print(f"Error decoding metadata JSON: {e}")
        return {}

def write_image_metadata(pil_image: Image.Image, params_dict: dict) -> Image.Image:
    """Creates a PngInfo object with the given parameters and attaches it to the image."""
    metadata = PngInfo()
    metadata.add_text("parameters", json.dumps(params_dict, indent=4)) # Added indent for readability
    # NOTE: This function's behavior (assigning to .info) is handled by a
    # corresponding fix in the event_handlers.prepare_image_for_download function.
    pil_image.info = metadata
    return pil_image

# --- UI Handler Functions ---

def open_and_check_metadata(temp_filepath: str):
    """
    Opens an image from a filepath string, converts to PIL, checks for metadata,
    and returns the PIL image, prompt, and full metadata dict.
    """
    if not temp_filepath:
        return None, "", {}

    try:
        # CORRECTED: Use the filepath string directly instead of trying to access a '.name' attribute.
        pil_image = Image.open(temp_filepath)
        # Ensure image is in a compatible mode (e.g., convert palette images)
        if pil_image.mode == 'P':
            pil_image = pil_image.convert('RGBA')
        else: # Ensure it has an alpha channel for consistency if it's RGB
            pil_image = pil_image.convert('RGBA')

        extracted_metadata = extract_metadata_from_pil_image(pil_image)
        prompt_preview = ""

        if extracted_metadata and any(key in extracted_metadata for key in shared_state.CREATIVE_PARAM_KEYS):
            prompt_preview = extracted_metadata.get('prompt', '')

        return pil_image, prompt_preview, extracted_metadata
    except Exception as e:
        # This will now only catch legitimate file opening/processing errors.
        print(f"Error processing uploaded file: {e}")
        gr.Warning(f"Could not open image. It may be corrupt or an unsupported format. Error: {e}")
        return None, "", {}

def ui_load_params_from_image_metadata(extracted_metadata: dict) -> list:
    """
    Loads creative parameters from a metadata dictionary and returns UI updates.
    """
    # This maps the canonical parameter names (e.g., 'steps') to their UI component keys (e.g., 'steps_ui').
    param_to_ui_map = {v: k for k, v in shared_state.UI_TO_WORKER_PARAM_MAP.items()}
    
    # Create a dictionary to hold updates, keyed by the UI component key.
    updates_dict = {}
    if extracted_metadata:
        gr.Info("Applying creative settings from image...")
        for param_key, value in extracted_metadata.items():
            if param_key in param_to_ui_map:
                ui_key = param_to_ui_map[param_key]
                if ui_key in shared_state.CREATIVE_UI_KEYS:
                    # Special handling for the variable CFG radio button
                    if ui_key == 'gs_schedule_shape_ui':
                        updates_dict[ui_key] = gr.update(value="Linear" if value else "Off")
                    else:
                        updates_dict[ui_key] = gr.update(value=value)

    # Construct the final list of updates in the correct order.
    final_updates = [updates_dict.get(key, gr.update()) for key in shared_state.CREATIVE_UI_KEYS]
    return final_updates

def create_params_from_ui(ui_keys: list, ui_values: tuple) -> dict:
    """
    Takes creative UI keys and values and returns a dictionary
    mapping the canonical parameter names to their current values.
    """
    ui_dict = dict(zip(ui_keys, ui_values))
    params_dict = {}
    for ui_key, value in ui_dict.items():
        worker_key = shared_state.UI_TO_WORKER_PARAM_MAP.get(ui_key)
        if worker_key:
             # Handle the special case for the gs_schedule radio button
            if ui_key == 'gs_schedule_shape_ui':
                params_dict[worker_key] = (value != 'Off')
            else:
                params_dict[worker_key] = value
    return params_dict