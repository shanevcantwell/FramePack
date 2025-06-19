# ui/event_handlers.py
import gradio as gr
import time
import tempfile
from PIL import Image

from . import metadata as metadata_manager
from . import shared_state
from . import queue as queue_manager
from . import workspace as workspace_manager

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

def prepare_image_for_download(pil_image, app_state, ui_keys, *creative_values):
    """Injects metadata, including LoRA settings from app_state, into the current image."""
    if not isinstance(pil_image, Image.Image):
        gr.Warning("No valid image to download.")
        return None

    image_copy = pil_image.copy()

    # Create the base dictionary from standard creative UI controls
    params_dict = metadata_manager.create_params_from_ui(ui_keys, creative_values)

    # --- ADDED: Extract LoRA state and add it to the params dictionary ---
    lora_state = app_state.get('lora_state', {})
    loaded_loras = lora_state.get('loaded_loras', {})

    if loaded_loras:
        # Create a simple dict of {lora_name: weight} for metadata
        lora_weights_for_save = {
            name: data.get("weight", 1.0)
            for name, data in loaded_loras.items()
        }
        params_dict['loras'] = lora_weights_for_save
    # --- End of added block ---

    image_with_metadata = metadata_manager.write_image_metadata(image_copy, params_dict)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        image_with_metadata.save(tmp_file.name, "PNG")
        gr.Info("Image with current settings (including LoRAs) prepared for download.")
        return gr.update(value=tmp_file.name)

def update_button_states(app_state, input_image_pil, queue_df_data):
    """
    Updates the 'Add Task' and 'Process Queue' button states based on input image
    and queue content.
    """
    queue_state = queue_manager.get_queue_state(app_state) # Uses imported helper
    is_processing = queue_state.get("processing", False)
    queue_has_tasks = len(queue_state["queue"]) > 0

    # Add Task to Queue button
    add_task_variant = "primary" if input_image_pil is not None else "secondary"

    # Process Queue button
    process_queue_interactive = False
    process_queue_variant = "secondary" # Default for empty/disabled

    if is_processing:
        process_queue_interactive = False
        process_queue_variant = "primary" # Already running, so it's "active" color
    elif queue_has_tasks:
        process_queue_interactive = True
        process_queue_variant = "primary" # Tasks present, ready to process
    else:
        process_queue_interactive = False
        process_queue_variant = "secondary" # No tasks, disabled

    return (
        gr.update(variant=add_task_variant),
        gr.update(interactive=process_queue_interactive, variant=process_queue_variant)
    )
