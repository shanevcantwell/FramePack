 # ui/metadata.py
import json
from PIL import Image
from PIL.PngImagePlugin import PngInfo

def extract_metadata_from_pil_image(pil_image: Image.Image) -> dict:
    """Extracts a 'parameters' dictionary from a PIL image's text chunk."""
    if pil_image is None: 
        return {}
    
    pnginfo_data = getattr(pil_image, 'text', None)
    if not isinstance(pnginfo_data, dict):
        return {}
    
    params_json_str = pnginfo_data.get('parameters')
    if not params_json_str:
        print("Metadata corrupted or not present in the image.")
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
    # This doesn't save the image, just attaches the metadata object for a later save call.
    pil_image.info = metadata.chunks
    return pil_image