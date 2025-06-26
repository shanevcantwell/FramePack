# ui/event_handlers.py
import gradio as gr
import time
import tempfile
from PIL import Image, PngImagePlugin

from . import metadata as metadata_manager
from . import shared_state as shared_state_module
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
        total_frames = int(total_seconds_ui * fps_ui)
        frames_per_segment = latent_window_size_ui * 4 - 3
        total_segments = int(max(round(total_frames / frames_per_segment), 1)) if frames_per_segment > 0 else 1
        return f"Calculated: {total_segments} Segments, {total_frames} Total Frames"
    except (TypeError, ValueError):
        return "Segments: Invalid input"

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
        return (
            gr.update(visible=True, value=None),    # IMAGE_FILE_INPUT_UI
            gr.update(visible=False, value=None),   # INPUT_IMAGE_DISPLAY_UI
            gr.update(interactive=False),           # CLEAR_IMAGE_BUTTON_UI
            gr.update(interactive=False),           # DOWNLOAD_IMAGE_BUTTON_UI
            gr.update(variant="secondary"),         # ADD_TASK_BUTTON
            "", {}, None
        )

    pil_image, prompt_preview, params = metadata_manager.open_and_check_metadata(filepath)

    has_loadable_metadata = bool(params and any(key in params for key in shared_state_module.CREATIVE_PARAM_KEYS))

    trigger_value = str(time.time()) if has_loadable_metadata else None

    final_prompt_preview = prompt_preview if has_loadable_metadata else ""

    return (gr.update(visible=False, value=None), gr.update(visible=True, value=pil_image), gr.update(interactive=True),
            gr.update(interactive=True), gr.update(variant="primary"), final_prompt_preview, params, trigger_value)

def clear_image_action():
    """Clears the input image and resets associated UI components."""
    return (
        gr.update(value=None, visible=True),
        gr.update(visible=False, value=None),
        gr.update(interactive=False, variant="secondary"), # CLEAR_IMAGE_BUTTON_UI
        gr.update(interactive=False, variant="secondary"), # DOWNLOAD_IMAGE_BUTTON_UI
        gr.update(variant="secondary"), # ADD_TASK_BUTTON
        {} # For extracted_metadata_state
    )

def prepare_image_for_download(pil_image, app_state, ui_keys, *creative_values):
    """Injects metadata into the current image and prepares it for download."""
    if not isinstance(pil_image, Image.Image):
        gr.Warning("No valid image to download.")
        return None

    image_copy = pil_image.copy()
    params_dict = metadata_manager.create_params_from_ui(ui_keys, creative_values) # ui_keys is passed in, so it's fine

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
    """
    Updates the interactive and variant states of all major control buttons
    based on the current application state. This function is the single source of
    truth for button states and is called after any action that might change them.
    """
    queue_state = get_queue_state(app_state)
    is_processing = queue_state.get("processing", False)
    queue = queue_state.get("queue", [])
    queue_has_tasks = bool(queue)
    has_image = input_image_pil is not None

    # Check for pending tasks, which controls the "Clear Pending" button.
    has_pending_tasks = any(task.get("status", "pending") == "pending" for task in queue)

    # --- Button Logic ---

    # Add Task, Clear Image, Download Image Buttons: Active if an image is present and not processing.
    image_actions_interactive = has_image
    add_task_variant = "primary" if image_actions_interactive else "secondary"

    # Process Queue Button: Active only if there are tasks and we are NOT processing.
    process_queue_interactive = queue_has_tasks and not is_processing
    process_queue_variant = "primary" if process_queue_interactive else "secondary"

    # Stop Processing Button: Active only when processing is underway.
    stop_processing_interactive = is_processing

    # Create Preview Button: Active only during processing, disabled if a preview is scheduled.
    # This now correctly checks the shared flag that the worker uses.
    create_preview_interactive = is_processing and not shared_state_module.shared_state_instance.preview_request_flag.is_set()
    create_preview_variant = "primary" if create_preview_interactive else "secondary" # This determines color (green if active, grey if disabled)

    # Save Queue Button: Active and green if there are tasks in the queue.
    save_queue_interactive = queue_has_tasks
    save_queue_variant = "primary" if save_queue_interactive else "secondary"

    # Clear Pending Button: Active if there are pending tasks and not processing.
    clear_pending_interactive = has_pending_tasks
    clear_queue_variant = "stop" if clear_pending_interactive else "secondary"

    return (
        gr.update(interactive=image_actions_interactive, variant=add_task_variant),      # ADD_TASK_BUTTON
        gr.update(interactive=process_queue_interactive, variant=process_queue_variant), # PROCESS_QUEUE_BUTTON
        gr.update(interactive=create_preview_interactive, variant=create_preview_variant), # CREATE_PREVIEW_BUTTON
        gr.update(interactive=image_actions_interactive, variant="secondary"), # CLEAR_IMAGE_BUTTON_UI
        gr.update(interactive=image_actions_interactive, variant="secondary"), # DOWNLOAD_IMAGE_BUTTON_UI
        gr.update(interactive=save_queue_interactive, variant=save_queue_variant),       # SAVE_QUEUE_BUTTON_UI
        gr.update(interactive=clear_pending_interactive, variant=clear_queue_variant),   # CLEAR_QUEUE_BUTTON_UI
        gr.update(interactive=stop_processing_interactive),                              # STOP_PROCESSING_BUTTON_UI
    )