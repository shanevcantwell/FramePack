# ui/workspace.py
# Contains functions for saving, loading, and managing workspace settings from files.

import gradio as gr
import json
import os
import traceback
import tempfile

from . import shared_state
from . import metadata as metadata_manager
from . import legacy_support  # <-- ADD THIS IMPORT
from PIL import Image

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
        'prompt_ui': '',
        'n_prompt_ui': '',
        'total_second_length_ui': 5.0,
        'seed_ui': -1,
        'preview_frequency_ui': 5,
        'segments_to_decode_csv_ui': '',
        'gs_ui': 10.0,
        'gs_schedule_shape_ui': 'Off',
        'gs_final_ui': 10.0,
        'roll_off_start_ui': 75,
        # 'roll_off_start_ui': 75, # MODIFIED: Removed duplicate key
        'roll_off_factor_ui': 1.0,
        'steps_ui': 25,
        'cfg_ui': 1.0,
        'rs_ui': 0.0,
        'use_teacache_ui': True,
        'use_fp32_transformer_output_checkbox_ui': False,
        'gpu_memory_preservation_ui': 6.0,
        'mp4_crf_ui': 18,
        'output_folder_ui_ctrl': outputs_folder,
        'latent_window_size_ui': 9,
    }

def save_settings_to_file(filepath, settings_dict):
    """Saves a dictionary of settings to a specified JSON file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(settings_dict, f, indent=4)
        gr.Info(f"Workspace saved to {filepath}")
    except Exception as e:
        gr.Warning(f"Error saving workspace: {e}")
        traceback.print_exc()

def load_settings_from_file(filepath, return_updates=True):
    """Loads settings from a JSON file and returns Gradio updates or raw values."""
    default_values = get_default_values_map()
    loaded_settings = {}

    if filepath and os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                loaded_settings = json.load(f)

            # MODIFIED: Apply legacy parameter conversion after loading.
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

    for key_enum in shared_state.ALL_TASK_UI_KEYS:
        key = key_enum.value
        value = final_settings.get(key, default_values.get(key))
        try:
            if key in ['seed_ui', 'latent_window_size_ui', 'steps_ui', 'mp4_crf_ui', 'preview_frequency_ui', 'roll_off_start_ui']:
                value = int(value)
            elif key in ['total_second_length_ui', 'cfg_ui', 'gs_ui', 'rs_ui', 'gpu_memory_preservation_ui', 'gs_final_ui', 'roll_off_factor_ui']:
                value = float(value)
            elif key in ['use_teacache_ui', 'use_fp32_transformer_output_checkbox_ui']:
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
            if 'output_folder_ui_ctrl' in settings:
                return os.path.expanduser(settings['output_folder_ui_ctrl'])
        except Exception as e:
            print(f"Warning: Could not load 'output_folder_ui_ctrl' from {filename_to_check}: {e}")
            traceback.print_exc()

    return default_output_folder_path

# --- UI Handler Functions ---

def save_workspace(*ui_values_tuple):
    """Prepares the full workspace settings as a JSON string for download."""
    ui_keys_list = [key.value for key in shared_state.ALL_TASK_UI_KEYS]
    settings_to_save = dict(zip(ui_keys_list, ui_values_tuple))
    json_data = json.dumps(settings_to_save, indent=4)

    try:
        temp_file_path = os.path.join(tempfile.gettempdir(), "goan_workspace.json")
        with open(temp_file_path, 'w', encoding='utf-8') as f:
            f.write(json_data)
        gr.Info("Workspace prepared for download.")
        return temp_file_path
    except Exception as e:
        gr.Warning(f"Error preparing workspace for download: {e}")
        traceback.print_exc()
        return None

def save_as_default_workspace(*ui_values_tuple):
    """
    Saves the current UI settings as the default startup configuration.
    """
    ui_keys_list = [key.value for key in shared_state.ALL_TASK_UI_KEYS]
    settings_to_save = dict(zip(ui_keys_list, ui_values_tuple))
    save_settings_to_file(SETTINGS_FILENAME, settings_to_save)
    gr.Info(f"Default settings saved. Restart the application for changes to take effect.")
    return gr.update(visible=True), gr.update(visible=True)

def save_ui_and_image_for_refresh(*args_from_ui_controls_tuple):
    """Saves UI state and the current image to temporary files for session recovery."""
    pil_image = args_from_ui_controls_tuple[0]
    all_ui_values_tuple = args_from_ui_controls_tuple[1:]
    ui_keys_list = [key.value for key in shared_state.ALL_TASK_UI_KEYS]
    full_params_map = dict(zip(ui_keys_list, all_ui_values_tuple))
    settings_to_save = full_params_map.copy()

    if pil_image and isinstance(pil_image, Image.Image):
        try:
            creative_ui_keys = [key.value for key in shared_state.CREATIVE_UI_KEYS]
            creative_ui_values = [full_params_map.get(key) for key in creative_ui_keys]
            creative_params = metadata_manager.create_params_from_ui(shared_state.CREATIVE_UI_KEYS, creative_ui_values)
            pnginfo_obj = metadata_manager.create_pnginfo_obj(creative_params)

            # Save the refresh image to a known temporary location
            refresh_image_path = os.path.join(tempfile.gettempdir(), REFRESH_IMAGE_FILENAME)

            pil_image.save(refresh_image_path, "PNG", pnginfo=pnginfo_obj)

            settings_to_save["refresh_image_path"] = refresh_image_path
            gr.Info(f"UI state saved, image written for refresh to {refresh_image_path}")
        except Exception as e:
            print(f"Error saving refresh image: {e}"); traceback.print_exc()
            gr.Warning(f"Could not save refresh image: {e}.")
            if "refresh_image_path" in settings_to_save: del settings_to_save["refresh_image_path"]
    else:
        if "refresh_image_path" in settings_to_save: del settings_to_save["refresh_image_path"]

    save_settings_to_file(UNLOAD_SAVE_FILENAME, settings_to_save)

def load_workspace(uploaded_file):
    """Loads a workspace from an uploaded JSON file."""
    if uploaded_file is None or not hasattr(uploaded_file, 'name') or not os.path.exists(uploaded_file.name):
        gr.Warning("No valid file selected or uploaded.")
        ui_keys_list = [key.value for key in shared_state.ALL_TASK_UI_KEYS]
        return [gr.update()] * len(ui_keys_list)

    return load_settings_from_file(uploaded_file.name)

# --- REWRITTEN FUNCTION ---
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
            print(f"Could not read {UNLOAD_SAVE_FILENAME} to find refresh image: {e}")
            pass # Continue with the settings path anyway
    # Fall back to the default settings file if no unload save exists.
    elif os.path.exists(SETTINGS_FILENAME):
        settings_file_path = SETTINGS_FILENAME

    if settings_file_path:
        print(f"Found workspace file to load on startup: {settings_file_path}")
    else:
        print("No workspace file found. Using default values.")

    # Return exactly two values, which may be None.
    # The switchboard is designed to handle this.
    return settings_file_path, image_path_to_load

def load_image_from_path(image_path):
    """Loads an image from a given path and returns an update for the Image component."""
    if image_path and os.path.exists(image_path):
        try:
            pil_image = Image.open(image_path)
            # Clean up the temporary session-restore image after loading it.
            if REFRESH_IMAGE_FILENAME in os.path.basename(image_path) or tempfile.gettempdir() in os.path.abspath(image_path):
                 try:
                     os.remove(image_path)
                 except OSError as e:
                     print(f"Error deleting temporary refresh image '{image_path}': {e}")
            return gr.update(value=pil_image, visible=True)
        except Exception as e:
            print(f"Error loading refresh image from '{image_path}': {e}"); traceback.print_exc()
            gr.Warning(f"Could not restore image from previous session: {e}")
            return gr.update(value=None, visible=False)

    return gr.update(value=None, visible=False)