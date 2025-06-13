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
    pil_image.info = metadata # The .info attribute is the correct place to assign the PngInfo object
    return pil_image

# --- UI Handler Functions (Moved from demo_gradio_svc.py) ---

def handle_image_upload_for_metadata(gallery_pil_list):
    """
    Checks an uploaded image for metadata and shows a confirmation modal if found.
    This function is triggered by the 'upload' event of the image gallery.
    """
    if not gallery_pil_list:
        return gr.update(visible=False)
    
    # The gallery component returns a list of (image, name) tuples.
    pil_image = gallery_pil_list[0][0] if isinstance(gallery_pil_list[0], tuple) else gallery_pil_list[0]
    
    if isinstance(pil_image, Image.Image):
        extracted_metadata = extract_metadata_from_pil_image(pil_image)
        # Show the modal only if metadata exists and contains relevant keys.
        if extracted_metadata and any(key in extracted_metadata for key in shared_state.CREATIVE_PARAM_KEYS):
            return gr.update(visible=True)
            
    return gr.update(visible=False)

def ui_load_params_from_image_metadata(gallery_data_list):
    """
    Loads ONLY the creative parameters from image metadata and returns UI updates.
    """
    updates = [gr.update()] * len(shared_state.CREATIVE_PARAM_KEYS)
    if not gallery_data_list:
        return updates
        
    pil_image = gallery_data_list[0][0] if isinstance(gallery_data_list[0], tuple) else gallery_data_list[0]
    extracted_metadata = extract_metadata_from_pil_image(pil_image)
    
    if not extracted_metadata:
        gr.Info("No parameters found in image.")
        return updates
        
    gr.Info(f"Applying creative settings from image...")
    for i, key in enumerate(shared_state.CREATIVE_PARAM_KEYS):
        if key in extracted_metadata:
            # Return a Gradio update object for each changed parameter.
            updates[i] = gr.update(value=extracted_metadata[key])
            
    return updates

def apply_and_hide_modal(gallery_data_list):
    """
    A wrapper function that applies the metadata and then hides the confirmation modal.
    """
    # The first output hides the modal, the rest are the parameter updates.
    return [gr.update(visible=False)] + ui_load_params_from_image_metadata(gallery_data_list)
