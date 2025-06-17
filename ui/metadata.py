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
    metadata.add_text("parameters", json.dumps(params_dict))
    pil_image.info = metadata
    return pil_image

# --- UI Handler Functions ---

def open_and_check_metadata(temp_file):
    """
    Opens a temporary file object, converts to PIL, checks for metadata,
    and returns the PIL image, prompt, and full metadata dict.
    """
    if not temp_file:
        return None, "", {}

    try:
        pil_image = Image.open(temp_file.name)
        extracted_metadata = extract_metadata_from_pil_image(pil_image)
        prompt_preview = ""

        if extracted_metadata and any(key in extracted_metadata for key in shared_state.CREATIVE_PARAM_KEYS):
            prompt_preview = extracted_metadata.get('prompt', '')

        return pil_image, prompt_preview, extracted_metadata
    except Exception as e:
        print(f"Error processing uploaded file: {e}")
        return None, "", {}

def ui_load_params_from_image_metadata(extracted_metadata):
    """
    Loads ONLY the creative parameters from a metadata dictionary and returns UI updates.
    """
    updates = [gr.update()] * len(shared_state.CREATIVE_UI_KEYS)
    if not extracted_metadata:
        gr.Info("No parameters found to apply.")
        return updates

    gr.Info(f"Applying creative settings from image...")
    # Map creative param worker keys to their corresponding UI component keys
    param_to_ui_map = {v: k for k, v in shared_state.UI_TO_WORKER_PARAM_MAP.items()}

    for i, param_key in enumerate(shared_state.CREATIVE_PARAM_KEYS):
        if param_key in extracted_metadata:
            value = extracted_metadata[param_key]
            ui_key = param_to_ui_map.get(param_key)

            # CORRECTED: Special handling for the variable CFG radio button
            if ui_key == 'gs_schedule_shape_ui':
                updates[i] = gr.update(value="Linear" if value else "Off")
            else:
                updates[i] = gr.update(value=value)

    return updates
    
# CHANGED: New helper function to construct a params dict from the UI values.
def create_params_from_ui(*ui_values):
    """
    Takes all creative UI values as *args and returns a dictionary
    mapping the UI keys to their current values.
    """
    return dict(zip(shared_state.CREATIVE_PARAM_KEYS, ui_values))
