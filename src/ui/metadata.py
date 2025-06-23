# ui/metadata.py
import json
import gradio as gr
from PIL import Image
from PIL.PngImagePlugin import PngInfo

# Import shared state for access to parameter key lists
from . import shared_state
from .enums import ComponentKey as K

# --- Core Metadata Functions ---

def extract_metadata_from_pil_image(pil_image: Image.Image) -> dict:
    """Extracts a 'parameters' dictionary from a PIL image's text chunk or info dictionary."""
    if pil_image is None:
        print("DEBUG (metadata.py - extract_metadata): pil_image is None.")
        return {}
    
    # Prioritize 'text' for explicit PNG text chunks
    pnginfo_data = getattr(pil_image, 'text', None) 
    if pnginfo_data is None and pil_image.info: # If 'text' is not directly available, check 'info'
        pnginfo_data = pil_image.info
        print(f"DEBUG (metadata.py - extract_metadata): Using pil_image.info. Content: {pnginfo_data}")
    else:
        print(f"DEBUG (metadata.py - extract_metadata): Using pil_image.text. Content: {pnginfo_data}")


    if not isinstance(pnginfo_data, dict):
        print(f"DEBUG (metadata.py - extract_metadata): pnginfo_data is not a dictionary. Type: {type(pnginfo_data)}")
        return {}

    # The key 'parameters' should be directly in the dictionary
    params_json_str = pnginfo_data.get('parameters') 
    if not params_json_str:
        print(f"DEBUG (metadata.py - extract_metadata): 'parameters' key not found in metadata.")
        return {}
    
    try:
        extracted_params = json.loads(params_json_str) #
        if not isinstance(extracted_params, dict):
            print(f"DEBUG (metadata.py - extract_metadata): Decoded parameters is not a dict. Type: {type(extracted_params)}")
            return {}
        print(f"DEBUG (metadata.py - extract_metadata): Successfully extracted parameters: {extracted_params}")
        return extracted_params
    except json.JSONDecodeError as e:
        print(f"Error decoding metadata JSON: {e}") #
        return {}

def create_pnginfo_obj(params_dict: dict) -> PngInfo:
    """Creates a PngInfo object with the given parameters."""
    metadata = PngInfo()
    metadata.add_text("parameters", json.dumps(params_dict, indent=4))
    return metadata


# --- UI Handler Functions ---
def open_and_check_metadata(temp_filepath: str):
    """
    Opens an image from a filepath string, converts to PIL, checks for metadata,
    and returns the PIL image, prompt, and full metadata dict.
    """
    if not temp_filepath:
        return None, "", {}
    try:
        pil_image = Image.open(temp_filepath)
        if pil_image.mode == 'P':
            pil_image = pil_image.convert('RGBA')
        else:
            pil_image = pil_image.convert('RGBA')
        extracted_metadata = extract_metadata_from_pil_image(pil_image)
        print(f"DEBUG (metadata.py): Extracted metadata: {extracted_metadata}") # ADD THIS
        prompt_preview = ""
        if extracted_metadata and any(key in extracted_metadata for key in shared_state.CREATIVE_PARAM_KEYS):
            prompt_preview = extracted_metadata.get('prompt', '')
            print(f"DEBUG (metadata.py): Metadata detected, prompt preview: '{prompt_preview}'") # ADD THIS
        else:
            print(f"DEBUG (metadata.py): No relevant metadata found or extraction failed.") # ADD THIS
        return pil_image, prompt_preview, extracted_metadata
    except Exception as e:
        print(f"DEBUG (metadata.py): Error processing uploaded file in open_and_check_metadata: {e}") # ADD THIS
        gr.Warning(f"Could not open image. It may be corrupt or an unsupported format. Error: {e}")
        return None, "", {}

def ui_load_params_from_image_metadata(extracted_metadata: dict) -> list:
    """
    Loads creative parameters from a metadata dictionary, performing necessary
    type conversions, and returns UI updates. This mirrors the logic from
    workspace.py to fix the UI not updating.
    """
    param_to_ui_map = {v: k for k, v in shared_state.UI_TO_WORKER_PARAM_MAP.items()}
    updates_dict = {}
    
    if extracted_metadata:
        gr.Info("Applying creative settings from image...")
        for param_key, value in extracted_metadata.items():
            if param_key in param_to_ui_map:
                ui_key = param_to_ui_map[param_key]
                if ui_key in shared_state.CREATIVE_UI_KEYS:
                    original_value = value
                    try:
                        # --- ADDED: Robust type conversion for UI components ---
                        # Sliders expecting integers
                        if ui_key in ['seed_ui', 'steps_ui', 'preview_frequency_ui']:
                            # Use float() first to gracefully handle numbers like 25.0
                            value = int(float(value))
                        # Sliders expecting floats
                        elif ui_key in ['total_second_length_ui', 'cfg_ui', 'gs_ui', 'rs_ui', 'gs_final_ui']:
                            value = float(value)
                        # Radio button expecting a string from a list of choices
                        elif ui_key == 'gs_schedule_shape_ui':
                            # In JSON metadata, this is saved as a boolean (true/false)
                            value = "Linear" if value else "Off"
                        
                        # Update the dictionary with the correctly typed value
                        updates_dict[ui_key] = gr.update(value=value)

                    except (ValueError, TypeError):
                        gr.Warning(f"Invalid value '{original_value}' for {ui_key} in image metadata. Skipping.")
                        # If conversion fails, we skip this parameter and do not update it.
                        continue

    # Construct the final list of updates in the correct order, sending an
    # update only for the components we found in the metadata.
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
            if ui_key == K.GS_SCHEDULE_SHAPE_UI.value:
                params_dict[worker_key] = (value != 'Off')
            else:
                params_dict[worker_key] = value
    return params_dict