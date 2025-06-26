# ui/event_handlers.py
import gradio as gr
import os
import time
import tempfile
from PIL import Image, PngImagePlugin

from . import metadata as metadata_manager
from . import shared_state
from . import workspace as workspace_manager
from .queue_helpers import get_queue_state
from .queue import autosave_queue_on_exit_action


def safe_shutdown_action(app_state, *ui_values):
    """Performs all necessary save operations to prepare the app for a clean shutdown."""
    print("Performing safe shutdown saves...")
    # This call to autosave is correct as it comes from queue.py
    autosave_queue_on_exit_action(app_state)
    workspace_manager.save_ui_and_image_for_refresh(*ui_values)
    gr.Info("Queue and UI state saved. It is now safe to close the terminal.")

def ui_update_total_segments(total_seconds_ui, latent_window_size_ui, fps_ui):
    """Calculates and displays the number of segments based on video length."""
    try:
        total_segments = int(max(round((total_seconds_ui * fps_ui) / (latent_window_size_ui * 4)), 1))
        return f"Calculated: {total_segments} Segments, {int(total_seconds_ui * fps_ui)} Total Frames"
    except (TypeError, ValueError):
        return "Segments: Invalid input"

def _reset_image_ui_to_empty():
    """Returns a tuple of updates to reset the image-related UI to its initial empty state."""
    return (
        gr.update(visible=True, value=None),  # IMAGE_FILE_INPUT_UI: Reset to visible, no value
        gr.update(visible=False, value=None), # INPUT_IMAGE_DISPLAY_UI: Hide, no value
        gr.update(visible=False),             # CLEAR_IMAGE_BUTTON_UI: Hide
        gr.update(visible=False),             # DOWNLOAD_IMAGE_BUTTON_UI: Hide
        gr.update(variant="secondary"),       # ADD_TASK_BUTTON: Disable
        "",                                   # METADATA_PROMPT_PREVIEW_UI: Clear
        {},                                   # EXTRACTED_METADATA_STATE: Clear
        None                                  # MODAL_TRIGGER_BOX: Clear
    )

def process_upload_and_show_image(temp_file_data):
    """
    Robustly handles file uploads from a gr.File component, checks for metadata,
    and returns UI updates.
    """
    filepath = None
    if isinstance(temp_file_data, str):
        filepath = temp_file_data
    elif hasattr(temp_file_data, 'name'):
        filepath = temp_file_data.name
    elif isinstance(temp_file_data, dict):
        filepath = temp_file_data.get('path')

    if not filepath:
        return _reset_image_ui_to_empty()

    # Check file type before attempting to open as an image
    allowed_image_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif']
    is_image = any(filepath.lower().endswith(ext) for ext in allowed_image_extensions)

    if not is_image:
        filename = os.path.basename(filepath)
        if filepath.lower().endswith('.json'):
            gr.Warning(f"'{filename}' is a JSON file. To load a workspace, please use the 'Load Workspace' button.")
        else:
            gr.Warning(f"Unsupported file type: '{filename}'. Please upload a valid image file ({', '.join(allowed_image_extensions)}).")
        return _reset_image_ui_to_empty()

    pil_image, prompt_preview, params = metadata_manager.open_and_check_metadata(filepath)

    # If image processing failed (e.g., invalid file type, corrupt image), reset to empty state
    if pil_image is None:
        return _reset_image_ui_to_empty()

    has_loadable_metadata = bool(params and any(key in params for key in shared_state.CREATIVE_PARAM_KEYS))
    trigger_value = str(time.time()) if has_loadable_metadata else None
    final_prompt_preview = prompt_preview if has_loadable_metadata else ""

    return (gr.update(visible=False, value=None), gr.update(visible=True, value=pil_image), gr.update(visible=True),
            gr.update(visible=True), gr.update(variant="primary"), final_prompt_preview, params, trigger_value)

def clear_image_action():
    """Clears the input image and resets associated UI components."""
    return (
        gr.update(value=None, visible=True),
        gr.update(visible=False, value=None),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(variant="secondary"),
        {} # For extracted_metadata_state
    )

def prepare_image_for_download(pil_image, app_state, ui_keys, *creative_values):
    """Injects metadata into the current image and prepares it for download."""
    if not isinstance(pil_image, Image.Image):
        gr.Warning("No valid image to download.")
        return None

    image_copy = pil_image.copy()
    params_dict = metadata_manager.create_params_from_ui(ui_keys, creative_values)

    lora_state = app_state.get('lora_state', {})
    if lora_state and lora_state.get('loaded_loras'):
        params_dict['loras'] = {
            name: data.get("weight", 1.0)
            for name, data in lora_state['loaded_loras'].items()
        }

    pnginfo_obj = metadata_manager.create_pnginfo_obj(params_dict)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        pil_image.save(tmp_file.name, "PNG", pnginfo=pnginfo_obj)
        gr.Info("Image with current settings prepared for download.")
        return gr.update(value=tmp_file.name)

def update_button_states(app_state, input_image_pil, queue_df_data):
    """Updates button states based on application state."""
    # This call to get_queue_state is now correct as it's imported from queue_helpers.py
    queue_state = get_queue_state(app_state)
    queue = queue_state.get("queue", [])
    is_processing = queue_state.get("processing", False)
    preview_requested = queue_state.get("preview_requested", False)
    queue_has_tasks = bool(queue)
    has_image = input_image_pil is not None

    # Add Task button
    add_task_variant = "primary" if has_image else "secondary"

    # --- Create Preview button Logic ---
    # Default state: Grey and disabled
    create_preview_variant = "secondary"
    create_preview_interactive = False
    create_preview_text = "📸 Create Preview"

    if is_processing:
        if preview_requested:
            # Yellow is not a standard Gradio variant, so we use secondary (grey)
            # to indicate a disabled, pending state.
            create_preview_variant = "secondary"
            create_preview_interactive = False
            create_preview_text = "Creating Segment Preview..."
        else:
            create_preview_variant = "primary" # Green and active
            create_preview_interactive = True
            
    # Image-related buttons (Clear, Download)
    image_button_variant = "primary" if has_image else "secondary"
    image_button_interactive = has_image

    # Save Queue button
    save_queue_variant = "primary" if queue_has_tasks else "secondary"
    save_queue_interactive = queue_has_tasks

    # Clear Pending button
    has_pending_tasks = any(task.get("status", "pending") == "pending" for task in queue)
    clear_pending_variant = "stop" if has_pending_tasks else "secondary"
    clear_pending_interactive = has_pending_tasks
    
    # Logic for Process Queue button
    process_queue_text = "▶️ Start Queue"
    process_queue_variant = "primary"
    process_queue_interactive = True
    if is_processing:
        process_queue_text = "⏹️ Stop Queue"
        process_queue_variant = "stop"
    elif not queue_has_tasks:
        process_queue_text = "Queue Empty"
        process_queue_variant = "secondary"
        process_queue_interactive = False
    
    return (
        gr.update(variant=add_task_variant),
        gr.update(interactive=process_queue_interactive, value=process_queue_text, variant=process_queue_variant),
        gr.update(interactive=create_preview_interactive, variant=create_preview_variant, value=create_preview_text),
        gr.update(interactive=image_button_interactive, variant=image_button_variant),
        gr.update(interactive=image_button_interactive, variant=image_button_variant),
        gr.update(interactive=save_queue_interactive, variant=save_queue_variant),
        gr.update(interactive=clear_pending_interactive, variant=clear_pending_variant),
    )