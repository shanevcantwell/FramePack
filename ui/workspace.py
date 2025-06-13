# ui/workspace.py
# Contains functions for saving, loading, and managing workspace settings from files.

import gradio as gr
import json
import os
import traceback
import tkinter as tk
from tkinter import filedialog
from PIL import Image

# Import shared state and other managers
from . import shared_state
from . import metadata as metadata_manager

# --- Constants ---
# Define filenames and the default output folder at the module level.
# These were moved from demo_gradio_svc.py
outputs_folder = './outputs_svc/'
SETTINGS_FILENAME = "goan_settings.json"
UNLOAD_SAVE_FILENAME = "goan_unload_save.json"
REFRESH_IMAGE_FILENAME = "goan_refresh_image.png"

# --- Core Save/Load Logic ---

def get_default_values_map():
    """Returns a dictionary with the default values for all UI settings."""
    return {
        'prompt': '', 'n_prompt': '', 'total_second_length': 5.0, 'seed': -1,
        'use_teacache': True, 'preview_frequency_ui': 5, 'segments_to_decode_csv': '',
        'gs_ui': 10.0, 'gs_schedule_shape_ui': 'Off', 'gs_final_ui': 10.0, 'steps': 25,
        'cfg': 1.0, 'latent_window_size': 9, 'gpu_memory_preservation': 6.0,
        'use_fp32_transformer_output_ui': False, 'rs': 0.0, 'mp4_crf': 18,
        'output_folder_ui': outputs_folder,
    }

def save_settings_to_file(filepath, *ui_values_tuple):
    """Saves a tuple of UI values to a specified JSON file."""
    settings_to_save = dict(zip(shared_state.ALL_TASK_UI_KEYS, ui_values_tuple))
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(settings_to_save, f, indent=4)
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
        loaded_settings = {}
    
    final_settings = {**default_values, **loaded_settings}
    output_values = [final_settings.get(key, default_values.get(key)) for key in shared_state.ALL_TASK_UI_KEYS]

    # Type correction to prevent errors when loading from JSON
    for i, key in enumerate(shared_state.ALL_TASK_UI_KEYS):
        try:
            if key in ['seed', 'latent_window_size', 'steps', 'mp4_crf', 'preview_frequency_ui']:
                output_values[i] = int(output_values[i])
            elif key in ['total_second_length', 'cfg', 'gs_ui', 'rs', 'gpu_memory_preservation', 'gs_final_ui']:
                output_values[i] = float(output_values[i])
            elif key in ['use_teacache', 'use_fp32_transformer_output_ui']:
                output_values[i] = bool(output_values[i])
        except (ValueError, TypeError):
            output_values[i] = default_values.get(key)
            
    return [gr.update(value=v) for v in output_values] if return_updates else output_values

# --- UI Handler Functions ---

def save_workspace(*ui_values_tuple):
    """Opens a file dialog to save the full workspace settings."""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.asksaveasfilename(
        defaultextension=".json",
        initialfile="goan_workspace.json",
        filetypes=[("JSON files", "*.json")]
    )
    root.destroy()
    if file_path:
        save_settings_to_file(file_path, *ui_values_tuple)
    else:
        gr.Warning("Save cancelled.")

def save_as_default_workspace(*ui_values_tuple):
    """Saves the current UI settings as the default startup configuration."""
    gr.Info(f"Saving current settings as default to {SETTINGS_FILENAME}")
    save_settings_to_file(SETTINGS_FILENAME, *ui_values_tuple)

def save_ui_and_image_for_refresh(*args_from_ui_controls_tuple):
    """Saves UI state and the current image to temporary files for session recovery."""
    gallery_list = args_from_ui_controls_tuple[0]
    all_ui_values_tuple = args_from_ui_controls_tuple[1:]
    full_params_map = dict(zip(shared_state.ALL_TASK_UI_KEYS, all_ui_values_tuple))
    settings_to_save = full_params_map.copy()

    if gallery_list and isinstance(gallery_list[0], (tuple, Image.Image)):
        pil_image = gallery_list[0][0] if isinstance(gallery_list[0], tuple) else gallery_list[0]
        try:
            creative_params = {k: full_params_map.get(k) for k in shared_state.CREATIVE_PARAM_KEYS}
            pil_image = metadata_manager.write_image_metadata(pil_image, creative_params)
            
            # Use the output folder path from the UI settings
            output_folder_path = full_params_map.get('output_folder_ui', outputs_folder)
            refresh_image_path = os.path.join(output_folder_path, REFRESH_IMAGE_FILENAME)
            
            pil_image.save(refresh_image_path)
            settings_to_save["refresh_image_path"] = refresh_image_path
        except Exception as e:
            gr.Warning(f"Could not save refresh image: {e}")
            
    # Save all settings to the unload file
    save_settings_to_file(UNLOAD_SAVE_FILENAME, *settings_to_save.values())

def load_workspace():
    """Opens a file dialog to load a workspace from a JSON file."""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
    root.destroy()
    return load_settings_from_file(file_path) if file_path else [gr.update()] * len(shared_state.ALL_TASK_UI_KEYS)

def load_workspace_on_start():
    """
    Loads settings on app startup, prioritizing a temporary session file,
    then a default file, and finally falling back to hardcoded defaults.
    """
    image_path_to_load = None
    settings_file = None
    
    if os.path.exists(UNLOAD_SAVE_FILENAME):
        settings_file = UNLOAD_SAVE_FILENAME
        try:
            with open(settings_file, 'r') as f:
                settings = json.load(f)
            # Check if the saved image path exists
            if "refresh_image_path" in settings and os.path.exists(settings["refresh_image_path"]):
                image_path_to_load = settings["refresh_image_path"]
        except Exception:
            pass # Ignore errors reading the temp file
    elif os.path.exists(SETTINGS_FILENAME):
        settings_file = SETTINGS_FILENAME

    if settings_file:
        print(f"Loading workspace from {settings_file}")
        ui_updates = load_settings_from_file(settings_file)
        if image_path_to_load:
            gr.Info("Restoring UI state and image from previous session.")
        if settings_file == UNLOAD_SAVE_FILENAME:
            os.remove(UNLOAD_SAVE_FILENAME) # Clean up temp file
        return [image_path_to_load] + ui_updates

    print("No workspace file found. Using default values.")
    default_vals = get_default_values_map()
    return [None] + [default_vals[key] for key in shared_state.ALL_TASK_UI_KEYS]

def load_image_from_path(image_path):
    """Loads an image from a given path and deletes the temporary file."""
    if image_path and os.path.exists(image_path):
        try:
            # The gallery component expects a list of (image, name) tuples.
            return gr.update(value=[(Image.open(image_path), "refresh_image")])
        finally:
            os.remove(image_path) # Clean up the temp image
    return gr.update(value=None)
