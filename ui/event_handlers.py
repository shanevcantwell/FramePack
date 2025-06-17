# ui/event_handlers.py
import gradio as gr
import time
import tempfile
from PIL import Image

from . import metadata as metadata_manager
from . import shared_state

def safe_shutdown_action(app_state, *ui_values):
    """
    Performs all necessary save operations to prepare the app for a clean shutdown.
    Saves the queue and the current UI state.
    """
    print("Performing safe shutdown saves...")
    
    # 1. Save the current queue to the autosave file
    queue_manager.autosave_queue_on_exit_action(app_state)
    
    # 2. Save the current UI state to the refresh file
    workspace_manager.save_ui_and_image_for_refresh(*ui_values)
    
    # 3. Notify the user that it's safe to close
    gr.Info("Queue and UI state saved. It is now safe to close the terminal.")

def ui_update_total_segments(total_seconds_ui, latent_window_size_ui):
    """Calculates and displays the number of segments based on video length."""
    try:
        total_segments = int(max(round((total_seconds_ui * 30) / (latent_window_size_ui * 4)), 1))
        return f"Calculated Total Segments: {total_segments}"
    except (TypeError, ValueError):
        return "Segments: Invalid input"

def process_upload_and_show_image(temp_file):
    """Processes file upload, checks for metadata, and returns UI updates."""
    if not temp_file:
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(variant="secondary"), None, "", {}, None

    pil_image, prompt_preview, params = metadata_manager.open_and_check_metadata(temp_file)

    if pil_image is None:
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(variant="secondary"), None, "", {}, None
    
    has_metadata = bool(params)
    trigger_value = str(time.time()) if has_metadata else None
    
    return gr.update(visible=False), gr.update(visible=True, value=pil_image), gr.update(visible=True), gr.update(visible=True), gr.update(variant="primary"), pil_image, prompt_preview, params, trigger_value

def clear_image_action():
    """Clears the input image and resets associated UI components, preserving creative settings."""
    return [
        gr.update(value=None, visible=True),   # image_file_input_ui
        gr.update(visible=False, value=None),  # input_image_display_ui
        gr.update(visible=False),              # clear_image_button_ui
        gr.update(visible=False),              # download_image_button_ui
        gr.update(variant="secondary"),        # add_task_button
        None,                                  # input_image_display_ui (value state)
        {}                                     # extracted_metadata_state
    ]

def prepare_image_for_download(pil_image, ui_keys, *creative_values):
    """Injects metadata into the current image for downloading."""
    if not isinstance(pil_image, Image.Image):
        gr.Warning("No valid image to download.")
        return None
    
    image_copy = pil_image.copy()
    params_dict = metadata_manager.create_params_from_ui(ui_keys, creative_values)
    image_with_metadata = metadata_manager.write_image_metadata(image_copy, params_dict)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        image_with_metadata.save(tmp_file.name, "PNG")
        gr.Info("Image with current settings prepared for download.")
        return gr.update(value=tmp_file.name)