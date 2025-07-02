# ui/workspace.py
# Contains functions for saving, loading, and managing workspace settings from files.

import gradio as gr
import json
import os
import zipfile
import tempfile
import logging

from . import shared_state as shared_state_module
from . import metadata as metadata_manager
from .enums import ComponentKey as K
from typing import Optional, Dict, Any, List, Tuple
from . import legacy_support
from PIL import Image

logger = logging.getLogger(__name__)

# --- Constants ---
outputs_folder = './outputs/'
SETTINGS_FILENAME = "goan_settings.json"
UNLOAD_SAVE_FILENAME = "goan_unload_save.json"
REFRESH_IMAGE_FILENAME = "goan_refresh_image.png"

# --- Core Save/Load Logic ---
def get_default_values_map():
    """
    Returns a dictionary with the default values for all UI settings.
    """
    return {
        K.POSITIVE_PROMPT: '',
        K.NEGATIVE_PROMPT: '',
        K.VIDEO_LENGTH_SLIDER: 5.0,
        K.SEED: -1,
        K.PREVIEW_FREQUENCY_SLIDER: 5,
        K.PREVIEW_SPECIFIED_SEGMENTS_TEXTBOX: '',
        K.FPS_SLIDER: 30,
        K.DISTILLED_CFG_START_SLIDER: 10.0,
        K.VARIABLE_CFG_SHAPE_RADIO: 'Off',
        K.DISTILLED_CFG_END_SLIDER: 10.0,
        K.ROLL_OFF_START_SLIDER: 75,
        K.ROLL_OFF_FACTOR_SLIDER: 1.0,
        K.STEPS_SLIDER: 25,
        K.REAL_CFG_SLIDER: 1.0,
        K.GUIDANCE_RESCALE_SLIDER: 0.0,
        K.USE_TEACACHE_CHECKBOX: True,
        K.USE_FP32_TRANSFORMER_OUTPUT_CHECKBOX: False,
        K.GPU_MEMORY_PRESERVATION_SLIDER: 6.0,
        K.MP4_CRF_SLIDER: 18,
        K.OUTPUT_FOLDER_TEXTBOX: outputs_folder,
        K.LATENT_WINDOW_SIZE_SLIDER: 9,
    }

def save_settings_to_file(filepath, settings_dict):
    """Saves a dictionary of settings to a specified JSON file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(settings_dict, f, indent=4)
        gr.Info(f"Workspace saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving workspace: {e}", exc_info=True)

def _apply_settings_dict_to_ui(settings_dict: Dict[K, Any], return_updates=True) -> List[Any]:
    """
    Takes a dictionary of settings and returns Gradio updates.
    This is a refactored helper to be used by both workspace and resume loading.
    """
    default_values = get_default_values_map()
    final_settings = {**default_values, **settings_dict}
    output_values = []

    # Iterate over the keys from the defaults map to ensure all settings are handled.
    for key in default_values.keys():
        value = final_settings.get(key, default_values[key])
        try:
            # This block ensures that values from JSON (which are often strings)
            # are converted to the correct type for the UI components.
            if key in [K.SEED_UI, K.LATENT_WINDOW_SIZE_UI, K.STEPS_UI, K.MP4_CRF_UI, K.PREVIEW_FREQUENCY_UI, K.ROLL_OFF_START_UI, K.FPS_UI, K.AUTO_RESUME_FREQUENCY_UI, K.AUTO_RESUME_RETENTION_UI]:
                value = int(float(value))
            elif key in [K.TOTAL_SECOND_LENGTH_UI, K.CFG_UI, K.GS_UI, K.RS_UI, K.GPU_MEMORY_PRESERVATION_UI, K.GS_FINAL_UI, K.ROLL_OFF_FACTOR_UI]:
                value = float(value)
            elif key in [K.USE_TEACACHE_UI, K.USE_FP32_TRANSFORMER_OUTPUT_CHECKBOX_UI]:
                value = bool(value)
        except (ValueError, TypeError):
            value = default_values.get(key)
            gr.Warning(f"Invalid value for {key} in loaded settings. Reverting to default.")
        output_values.append(value)

    if return_updates:
        # This is for Gradio's `outputs` list
        return [gr.update(value=v) for v in output_values]
    else:
        # This is for direct use
        return output_values

def load_settings_from_file(filepath, return_updates=True):
    """Loads settings from a JSON file and returns Gradio updates or raw values."""
    default_values = get_default_values_map()
    loaded_settings = {}

    if filepath and os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                loaded_settings_str_keys = json.load(f)
                # Create a set of all valid string values from the ComponentKey enum for a robust check.
                valid_keys = {item.value for item in K}
                loaded_settings = {K(k): v for k, v in loaded_settings_str_keys.items() if k in valid_keys}

            legacy_support.convert_legacy_params(loaded_settings)
            gr.Info(f"Loaded workspace from {filepath}")
        except Exception as e:
            gr.Warning(f"Could not load workspace from {filepath}: {e}")
            # If loading fails, use an empty dict to fall back to defaults
            loaded_settings = {}
    elif filepath is not None:
        # This handles the case where a non-existent path is passed, but not None
        gr.Warning(f"Workspace file not found at: {filepath}")

    final_settings = {**default_values, **loaded_settings}
    output_values = []

    # Iterate over the keys from the defaults map to ensure all settings are handled.
    for key in default_values.keys():
        value = final_settings.get(key, default_values[key])
        try:
            if key in [
                K.SEED, K.LATENT_WINDOW_SIZE_SLIDER, K.STEPS_SLIDER, K.MP4_CRF_SLIDER,
                K.PREVIEW_FREQUENCY_SLIDER, K.ROLL_OFF_START_SLIDER, K.FPS_SLIDER
            ]:
                value = int(value)
            elif key in [
                K.VIDEO_LENGTH_SLIDER, K.REAL_CFG_SLIDER, K.DISTILLED_CFG_START_SLIDER,
                K.GUIDANCE_RESCALE_SLIDER, K.GPU_MEMORY_PRESERVATION_SLIDER,
                K.DISTILLED_CFG_END_SLIDER, K.ROLL_OFF_FACTOR_SLIDER
            ]:
                value = float(value)
            elif key in [K.USE_TEACACHE_CHECKBOX, K.USE_FP32_TRANSFORMER_OUTPUT_CHECKBOX]:
                value = bool(value)
        except (ValueError, TypeError):
            value = default_values.get(key)
            gr.Warning(f"Invalid value for {key} in loaded settings. Reverting to default.")
        output_values.append(value)

    return [gr.update(value=v) for v in output_values] if return_updates else output_values


def get_initial_output_folder_from_settings():
    """
    Attempts to load the 'output_folder_ui_ctrl' value from unload or default settings.
    """
    default_output_folder_path = outputs_folder
    filename_to_check = None
    if os.path.exists(UNLOAD_SAVE_FILENAME):
        filename_to_check = UNLOAD_SAVE_FILENAME
    elif os.path.exists(SETTINGS_FILENAME):
        filename_to_check = SETTINGS_FILENAME

    if filename_to_check:
        try:
            with open(filename_to_check, 'r', encoding='utf-8') as f:
                settings = json.load(f)
            # Use new key name for output folder
            if 'output_folder_textbox' in settings:
                return os.path.expanduser(settings['output_folder_textbox'])
            # Fallback for legacy key
            if 'output_folder_ui_ctrl' in settings:
                return os.path.expanduser(settings['output_folder_ui_ctrl'])
        except Exception as e:
            logger.warning(f"Could not load output folder from {filename_to_check}: {e}", exc_info=True)

    return default_output_folder_path

# --- UI Handler Functions ---
def handle_file_drop(temp_file_data: Any) -> Tuple:
    """
    Unified handler for files dropped on the main input. It intelligently
    determines if the file is an image or a .goan_resume package and
    processes it accordingly.
    """
    filepath = None
    if isinstance(temp_file_data, str):
        filepath = temp_file_data
    elif hasattr(temp_file_data, 'name'):
        filepath = temp_file_data.name

    num_workspace_outputs = len(get_default_values_map())
    # 9 standard outputs + all workspace outputs
    total_outputs = standard_outputs + num_workspace_outputs # this needs to be fixed to test the length of standard_outputs

    if not filepath:
        return (gr.update(),) * total_outputs # Return no-op updates for all outputs

    # --- Case 1: It's a resume file ---
    if filepath.endswith(('.goan_resume', '.zip')):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(filepath, 'r') as zf:
                    # Signature check
                    if 'params.json' not in zf.namelist() or 'latent_history.pt' not in zf.namelist():
                        raise ValueError("Not a valid resume file")

                    # Extract all files
                    zf.extractall(temp_dir)

                    # Load params.json
                    params_path = os.path.join(temp_dir, 'params.json')
                    with open(params_path, 'r', encoding='utf-8') as f:
                        params_dict = json.load(f)

                    # Convert string keys to enums for _apply_settings_dict_to_ui
                    valid_keys = {item.value for item in K}
                    settings_dict = {K(k): v for k, v in params_dict.items() if k in valid_keys}

                    # Apply settings to UI
                    ui_updates = _apply_settings_dict_to_ui(settings_dict, return_updates=True)

                    # Load source image if present
                    image_update = gr.update()
                    image_path = os.path.join(temp_dir, 'source_image.png')
                    if os.path.exists(image_path):
                        try:
                            pil_image = Image.open(image_path)
                            image_update = gr.update(value=pil_image, visible=True)
                        except Exception as e:
                            logger.warning(f"Could not load source image from resume file: {e}")
                            image_update = gr.update(visible=False)
                    else:
                        image_update = gr.update(visible=False)

                    # Save latent path to state (for resume)
                    latent_path = os.path.join(temp_dir, 'latent_history.pt')
                    latent_path_update = gr.update(value=latent_path)

                    # Compose the output tuple:
                    output_tuple = (image_update, latent_path_update) + tuple(ui_updates) + (gr.update(),) * (total_outputs - 2 - len(ui_updates))
                    return output_tuple

        except Exception as e:
            logger.error(f"Error loading resume file: {e}", exc_info=True)
            gr.Warning(f"Could not load resume file: {e}")
            return (gr.update(),) * total_outputs

    # --- Case 2: It's an image file ---
    try:
        pil_image = Image.open(filepath)
        image_update = gr.update(value=pil_image, visible=True)

        # Use shared metadata extraction
        metadata_dict = metadata_manager.extract_metadata_from_pil_image(pil_image)

        if metadata_dict:
            valid_keys = {item.value for item in K}
            settings_dict = {K(k): v for k, v in metadata_dict.items() if k in valid_keys}
            ui_updates = _apply_settings_dict_to_ui(settings_dict, return_updates=True)
            output_tuple = (image_update,) + tuple(ui_updates) + (gr.update(),) * (total_outputs - 1 - len(ui_updates))
            return output_tuple
        else:
            # No metadata: only update the image, leave all other fields unchanged
            return (image_update,) + (gr.update(),) * (total_outputs - 1)
    except Exception as e:
        logger.warning(f"Could not load image: {e}")
        # Clear the image box and leave all other fields unchanged
        image_update = gr.update(value=None, visible=False)
        return (image_update,) + (gr.update(),) * (total_outputs - 1)

def save_workspace(*ui_values_tuple):
    """Prepares the full workspace settings as a JSON string for download."""
    # Convert enum keys to strings for JSON serialization
    settings_to_save = {key.value: value for key, value in zip(shared_state_module.ALL_TASK_UI_KEYS, ui_values_tuple)}
    json_data = json.dumps(settings_to_save, indent=4)

    try:
        temp_file_path = os.path.join(tempfile.gettempdir(), "goan_workspace.json")
        with open(temp_file_path, 'w', encoding='utf-8') as f:
            f.write(json_data)
        gr.Info("Workspace prepared for download.")
        return temp_file_path
    except Exception as e:
        logger.error(f"Error preparing workspace for download: {e}", exc_info=True)
        return None

def save_as_default_workspace(*ui_values_tuple):
    """
    Saves the current UI settings as the default startup configuration.
    """
    # Use the keys from the defaults map as the single source of truth.
    # Convert enum keys to strings for JSON serialization
    settings_to_save = {key.value: value for key, value in zip(get_default_values_map().keys(), ui_values_tuple)}
    save_settings_to_file(SETTINGS_FILENAME, settings_to_save)
    gr.Info(f"Default settings saved. Restart the application for changes to take effect.")
    return gr.update(visible=True), gr.update(visible=True)

def save_ui_and_image_for_refresh(*args_from_ui_controls_tuple):
    """Saves UI state and the current image to temporary files for session recovery."""
    pil_image = args_from_ui_controls_tuple[0]
    all_ui_values_tuple = args_from_ui_controls_tuple[1:] # All UI values except the image
    # Create a map with Enum keys for consistency
    full_params_map = dict(zip(shared_state_module.ALL_TASK_UI_KEYS, all_ui_values_tuple))
    # Convert to string keys for JSON saving
    settings_to_save = {key.value: value for key, value in full_params_map.items()}

    if pil_image and isinstance(pil_image, Image.Image):
        try:
            creative_ui_values = [full_params_map.get(key) for key in shared_state_module.CREATIVE_UI_KEYS] # Get values from the full map
            creative_params = metadata_manager.create_params_from_ui(shared_state_module.CREATIVE_UI_KEYS, creative_ui_values)
            pnginfo_obj = metadata_manager.create_pnginfo_obj(creative_params)

            # Save the refresh image to a known temporary location
            refresh_image_path = os.path.join(tempfile.gettempdir(), REFRESH_IMAGE_FILENAME)

            pil_image.save(refresh_image_path, "PNG", pnginfo=pnginfo_obj)

            settings_to_save["refresh_image_path"] = refresh_image_path
            gr.Info(f"UI state saved, image written for refresh to {refresh_image_path}")
        except Exception as e:
            logger.error(f"Error saving refresh image: {e}", exc_info=True)
            gr.Warning(f"Could not save refresh image: {e}")
            if "refresh_image_path" in settings_to_save: del settings_to_save["refresh_image_path"]
    else:
        if "refresh_image_path" in settings_to_save: del settings_to_save["refresh_image_path"]

    save_settings_to_file(UNLOAD_SAVE_FILENAME, settings_to_save)

def load_workspace(uploaded_file):
    """Loads a workspace from an uploaded JSON file."""
    if uploaded_file is None or not hasattr(uploaded_file, 'name') or not os.path.exists(uploaded_file.name):
        gr.Warning("No valid file selected or uploaded.")
        return [gr.update()] * len(shared_state_module.ALL_TASK_UI_KEYS)

    return load_settings_from_file(uploaded_file.name)

def load_workspace_on_start():
    """
    Finds the settings and refresh image file paths for application startup.

    This function's only responsibility is to return the two file paths.
    It does NOT load the data or create UI updates. This is handled by
    subsequent functions in the switchboard's .then() chain.
    This fixes the "returned too many output values" warning.
    """
    settings_file_path = None
    image_path_to_load = None

    # Prioritize the "unload_save" file which stores the last session state.
    if os.path.exists(UNLOAD_SAVE_FILENAME):
        settings_file_path = UNLOAD_SAVE_FILENAME
        try:
            # We still need to open it to find the potential image path
            with open(settings_file_path, 'r', encoding='utf-8') as f:
                settings = json.load(f)
            # Check for a refresh image path within the settings
            if "refresh_image_path" in settings and os.path.exists(settings["refresh_image_path"]):
                image_path_to_load = settings["refresh_image_path"]
        except (IOError, json.JSONDecodeError) as e:
            logger.warning(f"Could not read {UNLOAD_SAVE_FILENAME} to find refresh image: {e}")
            pass # Continue with the settings path anyway
    # Fall back to the default settings file if no unload save exists.
    elif os.path.exists(SETTINGS_FILENAME):
        settings_file_path = SETTINGS_FILENAME

    if settings_file_path:
        logger.info(f"Found workspace file to load on startup: {settings_file_path}")
    else:
        logger.info("No workspace file found. Using default values.")

    # Return exactly two values, which may be None.
    # The switchboard is designed to handle this.
    return settings_file_path, image_path_to_load

def load_image_from_path(image_path):
    """Loads an image from a given path and returns updates for the image and button components."""
    pil_image = None
    if image_path and os.path.exists(image_path):
        try:
            pil_image = Image.open(image_path)
            # Clean up the temporary session-restore image after loading it.
            if REFRESH_IMAGE_FILENAME in os.path.basename(image_path) or tempfile.gettempdir() in os.path.abspath(image_path):
                try:
                    os.remove(image_path)
                except OSError as e:
                    logger.warning(f"Error deleting temporary refresh image '{image_path}': {e}")
        except Exception as e:
            logger.error(f"Error loading refresh image from '{image_path}': {e}", exc_info=True)
            gr.Warning(f"Could not restore image from previous session: {e}")
            pil_image = None

    has_image = pil_image is not None
    return (gr.update(value=pil_image, visible=has_image), gr.update(interactive=has_image),
            gr.update(interactive=has_image), gr.update(visible=not has_image))

def load_settings_from_json(filepath=SETTINGS_FILENAME):
    """
    Loads settings from a JSON file and returns as a dictionary.
    This is a simplified loader for direct JSON access, bypassing any UI updates.

    If the file doesn't exist, or an error occurs, returns an empty dictionary.
    """
    if not os.path.exists(filepath):
        return {}

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            loaded_settings_str_keys = json.load(f)
            # Create a set of all valid string values from the ComponentKey enum for a robust check.
            valid_keys = {item.value for item in K}
            loaded_settings = {K(k): v for k, v in loaded_settings_str_keys.items() if k in valid_keys}

        legacy_support.convert_legacy_params(loaded_settings)
        logger.info(f"Settings loaded from {filepath}")
        return loaded_settings
    except Exception as e:
        logger.error(f"Error loading settings from {filepath}: {e}", exc_info=True)
        return {}

    """
    Loads settings from a JSON file and returns as a dictionary.
    This is a simplified loader for direct JSON access, bypassing any UI updates.

    If the file doesn't exist, or an error occurs, returns an empty dictionary.
    """
    if not os.path.exists(filepath):
        return {}

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            loaded_settings_str_keys = json.load(f)
            # Create a set of all valid string values from the ComponentKey enum for a robust check.
            valid_keys = {item.value for item in K}
            loaded_settings = {K(k): v for k, v in loaded_settings_str_keys.items() if k in valid_keys}

        legacy_support.convert_legacy_params(loaded_settings)
        logger.info(f"Settings loaded from {filepath}")
        return loaded_settings
    except Exception as e:
        logger.error(f"Error loading settings from {filepath}: {e}", exc_info=True)
        return {}