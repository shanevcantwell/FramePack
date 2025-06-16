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

def get_initial_output_folder_from_settings():
    """
    Attempts to load the 'output_folder_ui' value from UNLOAD_SAVE_FILENAME or SETTINGS_FILENAME.
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
            if 'output_folder_ui' in settings:
                return os.path.expanduser(settings['output_folder_ui'])
        except Exception as e:
            print(f"Warning: Could not load 'output_folder_ui' from {filename_to_check}: {e}")
            traceback.print_exc()
    
    return default_output_folder_path

# --- UI Handler Functions ---

def save_workspace(*ui_values_tuple):
    """Opens a file dialog to save the full workspace settings."""
    root = tk.Tk(); root.withdraw()
    file_path = filedialog.asksaveasfilename(defaultextension=".json", initialfile="goan_workspace.json", filetypes=[("JSON files", "*.json")])
    root.destroy()
    if file_path: save_settings_to_file(file_path, *ui_values_tuple)
    else: gr.Warning("Save cancelled.")

def save_as_default_workspace(*ui_values_tuple):
    """Saves the current UI settings as the default startup configuration."""
    gr.Info(f"Saving current settings as default to {SETTINGS_FILENAME}")
    save_settings_to_file(SETTINGS_FILENAME, *ui_values_tuple)

def save_ui_and_image_for_refresh(*args_from_ui_controls_tuple):
    """Saves UI state and the current image to temporary files for session recovery."""
    pil_image = args_from_ui_controls_tuple[0]
    all_ui_values_tuple = args_from_ui_controls_tuple[1:]
    full_params_map = dict(zip(shared_state.ALL_TASK_UI_KEYS, all_ui_values_tuple))
    settings_to_save = full_params_map.copy()

    if pil_image and isinstance(pil_image, Image.Image):
        try:
            creative_params = {k: full_params_map.get(k) for k in shared_state.CREATIVE_PARAM_KEYS}
            pil_image_with_metadata = metadata_manager.write_image_metadata(pil_image, creative_params)
            output_folder_path = full_params_map.get('output_folder_ui', outputs_folder)
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

    save_settings_to_file(UNLOAD_SAVE_FILENAME, *settings_to_save.values())

def load_workspace():
    """Opens a file dialog to load a workspace from a JSON file."""
    root = tk.Tk(); root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
    root.destroy()
    return load_settings_from_file(file_path) if file_path else [gr.update()] * len(shared_state.ALL_TASK_UI_KEYS)

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
        ui_updates = load_settings_from_file(settings_file)
        if image_path_to_load: gr.Info("Restoring UI state and image from previous session.")
        if settings_file == UNLOAD_SAVE_FILENAME: os.remove(UNLOAD_SAVE_FILENAME)
        return [image_path_to_load] + ui_updates

    print("No workspace file found. Using default values.")
    default_vals = get_default_values_map()
    return [None] + [default_vals[key] for key in shared_state.ALL_TASK_UI_KEYS]

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
            try: os.remove(image_path)
            except OSError as e: print(f"Error deleting temporary refresh image '{image_path}': {e}")
    
    return gr.update(value=None, visible=False)