# ui/workspace.py
# Contains functions for saving, loading, and managing workspace settings from files.

import gradio as gr
import json
import os
import zipfile
import tempfile
import logging
import time
import pandas as pd # New: Import pandas for DataFrame handling

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
QUEUE_STATE_JSON_IN_ZIP = "queue_state.json" # Ensure this constant is available

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
            if key in [K.SEED, K.LATENT_WINDOW_SIZE_SLIDER, K.STEPS_SLIDER, K.MP4_CRF_SLIDER, K.PREVIEW_FREQUENCY_SLIDER, K.ROLL_OFF_START_SLIDER, K.FPS_SLIDER,
                       # K.AUTO_RESUME_FREQUENCY, K.AUTO_RESUME_RETENTION
                       ]:
                value = int(float(value))
            elif key in [K.VIDEO_LENGTH_SLIDER, K.REAL_CFG_SLIDER, K.DISTILLED_CFG_START_SLIDER, K.GUIDANCE_RESCALE_SLIDER, K.GPU_MEMORY_PRESERVATION_SLIDER, K.DISTILLED_CFG_END_SLIDER, K.ROLL_OFF_FACTOR_SLIDER]:
                value = float(value)
            elif key in [K.USE_TEACACHE_CHECKBOX, K.USE_FP32_TRANSFORMER_OUTPUT_CHECKBOX]:
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

    return _apply_settings_dict_to_ui(loaded_settings, return_updates)

def load_queue_from_zip_internal(zip_filepath):
    """Internal function to load queue state from a zip file."""
    queue_data = {"queue": [], "next_id": 1, "processing": False, "editing_task_id": None}
    try:
        with zipfile.ZipFile(zip_filepath, 'r') as zf:
            if QUEUE_STATE_JSON_IN_ZIP in zf.namelist():
                with zf.open(QUEUE_STATE_JSON_IN_ZIP) as f:
                    loaded_data = json.load(f)
                    if isinstance(loaded_data, dict):
                        queue_data.update(loaded_data)
                    logger.info(f"Queue loaded from {zip_filepath}")
            else:
                logger.warning(f"'{QUEUE_STATE_JSON_IN_ZIP}' not found in zip file.")
    except Exception as e:
        logger.error(f"Error loading queue from zip: {e}", exc_info=True)
        gr.Warning(f"Could not load queue from zip: {e}")
    return queue_data

def load_workspace_on_start() -> Tuple[Dict[str, Any], Optional[Image.Image], pd.DataFrame]:
    """
    Loads initial application state, including UI settings, last image, and queue.
    Returns a tuple of (app_state_dict, input_image_pil, queue_df).
    """
    app_state = {
        "queue_state": {"queue": [], "next_id": 1, "processing": False, "editing_task_id": None},
        "last_completed_video_path": None,
        "lora_state": {"loaded_loras": {}},
        "last_used_seed": -1 # Initialize last_used_seed
    }
    input_image_pil = None
    queue_df = pd.DataFrame(columns=["↑", "↓", "⏸️", "✎", "✖", "Status", "Prompt", "Image", "Length", "ID"])

    settings_file_path = None
    image_path_to_load = None

    # Prioritize the "unload_save" file which stores the last session state.
    if os.path.exists(UNLOAD_SAVE_FILENAME):
        settings_file_path = UNLOAD_SAVE_FILENAME
        try:
            with open(settings_file_path, 'r', encoding='utf-8') as f:
                settings = json.load(f)
            if "refresh_image_path" in settings and os.path.exists(settings["refresh_image_path"]):
                image_path_to_load = settings["refresh_image_path"]
            # Load last_used_seed from settings if available
            if 'last_used_seed' in settings:
                app_state['last_used_seed'] = settings['last_used_seed']
        except (IOError, json.JSONDecodeError) as e:
            logger.warning(f"Could not read {UNLOAD_SAVE_FILENAME} to find refresh image or last seed: {e}")
    # Fall back to the default settings file if no unload save exists.
    elif os.path.exists(SETTINGS_FILENAME):
        settings_file_path = SETTINGS_FILENAME
        try:
            with open(settings_file_path, 'r', encoding='utf-8') as f:
                settings = json.load(f)
            if 'last_used_seed' in settings:
                app_state['last_used_seed'] = settings['last_used_seed']
        except (IOError, json.JSONDecodeError) as e:
            logger.warning(f"Could not read {SETTINGS_FILENAME} to find last seed: {e}")


    # Load UI settings
    if settings_file_path:
        loaded_settings = load_settings_from_json(settings_file_path)
        # Update app_state with any relevant settings, e.g., output_folder
        if K.OUTPUT_FOLDER_TEXTBOX in loaded_settings:
            app_state['output_folder'] = os.path.expanduser(loaded_settings[K.OUTPUT_FOLDER_TEXTBOX])
        # Apply loaded settings to the default values map for UI components
        for key, value in loaded_settings.items():
            # Only update if the key is one of the ALL_TASK_UI_KEYS
            if key in shared_state_module.ALL_TASK_UI_KEYS:
                # This ensures that the initial UI components get their values
                # from the loaded settings. The actual Gradio components will be
                # updated by the .load event in switchboard_startup.
                pass # The actual update happens in switchboard_startup

    # Load refresh image
    if image_path_to_load and os.path.exists(image_path_to_load):
        try:
            input_image_pil = Image.open(image_path_to_load).convert('RGBA')
            # Clean up the temporary session-restore image after loading it.
            try:
                os.remove(image_path_to_load)
            except OSError as e:
                logger.warning(f"Error deleting temporary refresh image '{image_path_to_load}': {e}")
        except Exception as e:
            logger.error(f"Error loading refresh image from '{image_path_to_load}': {e}", exc_info=True)
            gr.Warning(f"Could not restore image from previous session: {e}")
            input_image_pil = None

    # Load autosaved queue
    autosave_queue_path = os.path.join(tempfile.gettempdir(), "goan_autosave_queue.zip")
    if os.path.exists(autosave_queue_path):
        queue_data = load_queue_from_zip_internal(autosave_queue_path)
        app_state['queue_state'].update(queue_data)
        queue_df = pd.DataFrame(app_state['queue_state']['queue'])
        logger.info("Autosaved queue loaded on startup.")

    logger.info("Initial workspace loading complete.")
    return app_state, input_image_pil, queue_df


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
# In ui/workspace.py

def handle_file_drop(temp_file_data: any) -> tuple:
    """
    Patched handler for file drops. This version returns a correctly
    sized tuple of updates to resolve the ValueError on startup.
    """
    # 1. Define the full list of output keys. This list must exactly match
    # the components used to build `upload_outputs` in switchboard_image.py.
    # This removes the "magic number" and fixes the crash.
    output_keys = [
        K.IMAGE_FILE_INPUT, K.INPUT_IMAGE_DISPLAY, K.CLEAR_IMAGE_BUTTON,
        K.DOWNLOAD_IMAGE_BUTTON, K.ADD_TASK_BUTTON, K.METADATA_PROMPT_PREVIEW,
        K.EXTRACTED_METADATA_STATE, K.METADATA_MODAL_TRIGGER_STATE
    ] + shared_state_module.CREATIVE_UI_KEYS

    # 2. Initialize a dictionary to hold all possible updates.
    updates = {key: gr.update() for key in output_keys}

    filepath = None
    if isinstance(temp_file_data, str):
        filepath = temp_file_data
    elif hasattr(temp_file_data, 'name'):
        filepath = temp_file_data.name

    if not filepath:
        # If no file, return default "no change" updates.
        return tuple(updates.get(key, gr.update()) for key in output_keys)

    # Block resume files for now, as requested.
    if filepath.endswith(('.goan_resume', '.zip')):
        gr.Warning("Resume file processing is currently disabled.")
        return tuple(updates.get(key, gr.update()) for key in output_keys)

    # Process the image file and populate the updates dictionary.
    try:
        pil_image = Image.open(filepath)
        updates[K.INPUT_IMAGE_DISPLAY] = gr.update(value=pil_image, visible=True)
        updates[K.IMAGE_FILE_INPUT] = gr.update(visible=False)

        params = metadata_manager.extract_metadata_from_pil_image(pil_image)
        if params:
            updates[K.EXTRACTED_METADATA_STATE] = params
            updates[K.METADATA_PROMPT_PREVIEW] = params.get('prompt', '')
            updates[K.METADATA_MODAL_TRIGGER_STATE] = gr.update(value=str(time.time()))
    except Exception as e:
        gr.Warning(f"Could not load file as an image: {e}")
        updates[K.INPUT_IMAGE_DISPLAY] = gr.update(value=None, visible=False)
        updates[K.IMAGE_FILE_INPUT] = gr.update(visible=True, value=None)

    # 3. Assemble the final tuple in the correct order.
    return tuple(updates.get(key, gr.update()) for key in output_keys)

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
    settings_to_save_with_enums = dict(zip(get_default_values_map().keys(), ui_values_tuple))

    # Exclude prompts and seed from the saved default settings.
    keys_to_exclude = {K.POSITIVE_PROMPT, K.NEGATIVE_PROMPT, K.SEED}
    for key in keys_to_exclude:
        if key in settings_to_save_with_enums:
            del settings_to_save_with_enums[key]

    settings_to_save = {key.value: value for key, value in settings_to_save_with_enums.items()}

    save_settings_to_file(SETTINGS_FILENAME, settings_to_save)
    gr.Info("Default settings saved. Restart the application for changes to take effect.")
    # Return a single update object for the single target component.
    return gr.update(visible=True)

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