# ui/workspace.py
# Contains functions for saving, loading, and managing workspace settings from files.

import gradio as gr
import json
import os
import traceback
import tempfile # ADDED: Import tempfile module
# REMOVED: tkinter is no longer used for file dialogs.
# import tkinter as tk
# from tkinter import filedialog
from PIL import Image

# Import shared state and other managers
from . import shared_state
from . import metadata as metadata_manager

# --- Constants ---
outputs_folder = './outputs_svc/'
SETTINGS_FILENAME = "goan_settings.json"
UNLOAD_SAVE_FILENAME = "goan_unload_save.json"
REFRESH_IMAGE_FILENAME = "goan_refresh_image.png"

# --- Core Save/Load Logic ---

def get_default_values_map():
    """
    Returns a dictionary with the default values for all UI settings.
    CHANGED: The keys are now the UI component keys (e.g., 'prompt_ui') to match
    the centralized keys in shared_state.py, fixing startup load errors.
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

def save_settings_to_file(filepath, settings_dict): # CHANGED: Now accepts a settings dictionary
    """Saves a dictionary of settings to a specified JSON file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(settings_dict, f, indent=4) # CHANGED: Dump the dictionary directly
        gr.Info(f"Workspace saved to {filepath}")
    except Exception as e:
        gr.Warning(f"Error saving workspace: {e}")
        traceback.print_exc()

def load_settings_from_file(filepath, return_updates=True):
    """Loads settings from a JSON file and returns Gradio updates or raw values."""
    default_values = get_default_values_map()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            loaded_settings = json.load(f)
        gr.Info(f"Loaded workspace from {filepath}")
    except Exception as e:
        gr.Warning(f"Could not load workspace from {filepath}: {e}")
        loaded_settings = {} # If loading fails, use an empty dict to fall back to defaults

    final_settings = {**default_values, **loaded_settings}
    output_values = [final_settings.get(key, default_values.get(key)) for key in shared_state.ALL_TASK_UI_KEYS]

    for i, key in enumerate(shared_state.ALL_TASK_UI_KEYS):
        try:
            # Note: This integer/float conversion logic might be brittle if key names change.
            # It relies on string matching within the key name.
            if key in ['seed_ui', 'latent_window_size_ui', 'steps_ui', 'mp4_crf_ui', 'preview_frequency_ui']:
                output_values[i] = int(output_values[i])
            elif key in ['total_second_length_ui', 'cfg_ui', 'gs_ui', 'rs_ui', 'gpu_memory_preservation_ui', 'gs_final_ui']:
                output_values[i] = float(output_values[i])
            elif key in ['use_teacache_ui', 'use_fp32_transformer_output_checkbox_ui']:
                output_values[i] = bool(output_values[i])
        except (ValueError, TypeError):
            # If conversion fails, revert to default for that specific key
            output_values[i] = default_values.get(key)
            gr.Warning(f"Invalid value for {key} in loaded settings. Reverting to default.") # Added a warning here
            
    return [gr.update(value=v) for v in output_values] if return_updates else output_values

def load_workspace(uploaded_file):
    """Loads a workspace from an uploaded JSON file."""
    if uploaded_file is None or not hasattr(uploaded_file, 'name') or not os.path.exists(uploaded_file.name):
        gr.Warning("No valid file selected or uploaded.")
        # If no valid file, return updates to default values
        return load_settings_from_file(None) # Pass None to trigger default loading path
        # return [gr.update()] * len(shared_state.ALL_TASK_UI_KEYS) # Original placeholder, less robust

    return load_settings_from_file(uploaded_file.name)

def get_initial_output_folder_from_settings():
    """
    Attempts to load the 'output_folder_ui_ctrl' value from UNLOAD_SAVE_FILENAME or SETTINGS_FILENAME.
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
            # CHANGED: Looks for the correct UI key for the output folder.
            if 'output_folder_ui_ctrl' in settings:
                return os.path.expanduser(settings['output_folder_ui_ctrl'])
        except Exception as e:
            print(f"Warning: Could not load 'output_folder_ui_ctrl' from {filename_to_check}: {e}")
            traceback.print_exc()

    return default_output_folder_path

# --- UI Handler Functions ---

# CHANGED: save_workspace now prepares data for download and returns a file path.
def save_workspace(*ui_values_tuple):
    """Prepares the full workspace settings as a JSON string for download."""
    settings_to_save = dict(zip(shared_state.ALL_TASK_UI_KEYS, ui_values_tuple))
    json_data = json.dumps(settings_to_save, indent=4)

    # Create a temporary file to hold the JSON data for Gradio to download
    # Gradio's DownloadButton expects a path to a file.
    try:
        temp_file_path = os.path.join(tempfile.gettempdir(), "goan_workspace.json")
        with open(temp_file_path, 'w', encoding='utf-8') as f:
            f.write(json_data)
        gr.Info("Workspace prepared for download.")
        return temp_file_path # Return the path for the DownloadButton
    except Exception as e:
        gr.Warning(f"Error preparing workspace for download: {e}")
        traceback.print_exc()
        return None # Return None if preparation fails

def save_as_default_workspace(*ui_values_tuple):
    """
    Saves the current UI settings as the default startup configuration and
    returns updates to show the relaunch notification.
    """
    settings_to_save = dict(zip(shared_state.ALL_TASK_UI_KEYS, ui_values_tuple))
    new_output_folder = settings_to_save.get('output_folder_ui_ctrl')

    # Load existing settings to append the new allowed path.
    try:
        with open(SETTINGS_FILENAME, 'r', encoding='utf-8') as f:
            existing_settings = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_settings = {}

    # Add the new path to a persistent list of allowed paths.
    user_allowed_paths = existing_settings.get('user_allowed_paths', [])
    if new_output_folder and new_output_folder not in user_allowed_paths:
        user_allowed_paths.append(new_output_folder)

    # Combine the UI settings with the updated path list and save.
    settings_to_save['user_allowed_paths'] = user_allowed_paths
    # CHANGED: Pass the settings_to_save dictionary directly.
    save_settings_to_file(SETTINGS_FILENAME, settings_to_save)

    gr.Info(f"Default settings saved. Restart the application for the new output path to be allowed.")

    # Return updates to make the relaunch UI visible.
    return gr.update(visible=True), gr.update(visible=True)

def save_ui_and_image_for_refresh(*args_from_ui_controls_tuple):
    """Saves UI state and the current image to temporary files for session recovery."""
    pil_image = args_from_ui_controls_tuple[0]
    all_ui_values_tuple = args_from_ui_controls_tuple[1:]
    full_params_map = dict(zip(shared_state.ALL_TASK_UI_KEYS, all_ui_values_tuple))
    settings_to_save = full_params_map.copy()

    if pil_image and isinstance(pil_image, Image.Image):
        try:
            # Use the robust metadata creation function
            creative_ui_values = [full_params_map.get(key) for key in shared_state.CREATIVE_UI_KEYS]
            creative_params = metadata_manager.create_params_from_ui(shared_state.CREATIVE_UI_KEYS, creative_ui_values)

            pil_image_with_metadata = metadata_manager.write_image_metadata(pil_image, creative_params)
            output_folder_path = full_params_map.get('output_folder_ui_ctrl', outputs_folder)
            refresh_image_path = os.path.join(output_folder_path, REFRESH_IMAGE_FILENAME)
            os.makedirs(output_folder_path, exist_ok=True)
            pil_image_with_metadata.save(refresh_image_path)
            settings_to_save["refresh_image_path"] = refresh_image_path
            gr.Info(f"UI state saved, image written for refresh to {refresh_image_path}")
        except Exception as e:
            print(f"Error saving refresh image: {e}"); traceback.print_exc()
            gr.Warning(f"Could not save refresh image: {e}.")
            if "refresh_image_path" in settings_to_save: del settings_to_save["refresh_image_path"]
    else:
        if "refresh_image_path" in settings_to_save: del settings_to_save["refresh_image_path"]

    # CHANGED: Pass the settings_to_save dictionary directly.
    save_settings_to_file(UNLOAD_SAVE_FILENAME, settings_to_save)

# CHANGED: load_workspace now accepts an uploaded file object.
def load_workspace(uploaded_file):
    """Loads a workspace from an uploaded JSON file."""
    if uploaded_file is None or not hasattr(uploaded_file, 'name') or not os.path.exists(uploaded_file.name):
        gr.Warning("No valid file selected or uploaded.")
        # Return updates that effectively do nothing or reset to defaults if needed
        return [gr.update()] * len(shared_state.ALL_TASK_UI_KEYS)

    # Call the core load logic with the path of the uploaded file
    # Gradio's UploadButton provides a temporary file path in `uploaded_file.name`.
    return load_settings_from_file(uploaded_file.name)

def load_workspace_on_start():
    """Loads settings on app startup."""
    image_path_to_load = None; settings_file = None
    if os.path.exists(UNLOAD_SAVE_FILENAME):
        settings_file = UNLOAD_SAVE_FILENAME
        try:
            with open(settings_file, 'r') as f: settings = json.load(f)
            if "refresh_image_path" in settings and os.path.exists(settings["refresh_image_path"]):
                image_path_to_load = settings["refresh_image_path"]
        except Exception: pass
    elif os.path.exists(SETTINGS_FILENAME): settings_file = SETTINGS_FILENAME

    if settings_file:
        print(f"Loading workspace from {settings_file}")
        # When loading settings from disk, we pass the file path and let it handle reading.
        ui_updates = load_settings_from_file(settings_file)
        if image_path_to_load: gr.Info("Restoring UI state and image from previous session.")
        # Ensure the temporary unload save file is removed after use
        if settings_file == UNLOAD_SAVE_FILENAME:
            try:
                os.remove(UNLOAD_SAVE_FILENAME)
            except OSError as e:
                print(f"Error deleting temporary unload save file '{UNLOAD_SAVE_FILENAME}': {e}")
        return [image_path_to_load] + ui_updates

    print("No workspace file found. Using default values.")
    default_vals = get_default_values_map()
    return [None] + [default_vals.get(key) for key in shared_state.ALL_TASK_UI_KEYS]

def load_image_from_path(image_path):
    """Loads an image from a given path and returns an update for the Image component."""
    # CHANGED: Returns visibility updates to ensure consistent state on load.
    if image_path and os.path.exists(image_path):
        try:
            pil_image = Image.open(image_path)
            return gr.update(value=pil_image, visible=True)
        except Exception as e:
            print(f"Error loading refresh image from '{image_path}': {e}"); traceback.print_exc()
            gr.Warning(f"Could not restore image from previous session: {e}")
            return gr.update(value=None, visible=False)
        finally:
            # Only remove if it's explicitly marked as a temporary refresh image
            # This check prevents accidental deletion of user's own image files
            if REFRESH_IMAGE_FILENAME in os.path.basename(image_path):
                try: os.remove(image_path)
                except OSError as e: print(f"Error deleting temporary refresh image '{image_path}': {e}")

    return gr.update(value=None, visible=False)