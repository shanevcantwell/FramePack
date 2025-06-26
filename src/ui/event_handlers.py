# ui/event_handlers.py
import gradio as gr
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

    has_loadable_metadata = bool(params and any(key in params for key in shared_state.CREATIVE_PARAM_KEYS))

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
    """
    Updates the interactive and variant states of all major control buttons
    based on the current application state. This function is the single source of
    truth for button states and is called after any action that might change them.
    """
    queue_state = get_queue_state(app_state)
    is_processing = queue_state.get("processing", False)
    # The value of a gr.DataFrame is a list of lists. An empty queue is an empty list.
    queue_has_tasks = bool(queue_df_data)
    has_image = input_image_pil is not None

    # Add Task Button: Interactive and primary if an image is present.
    add_task_interactive = has_image
    add_task_variant = "primary" if has_image else "secondary"

    # Process Queue Button: Interactive if queue has tasks and not processing.
    process_queue_interactive = not is_processing and queue_has_tasks
    process_queue_variant = "primary" if queue_has_tasks else "secondary"

    # Abort/Preview Button: Interactive only when processing.
    abort_task_interactive = is_processing

    # Clear/Download Image Buttons: Interactive if an image is present and not processing.
    image_buttons_interactive = has_image and not is_processing

    # Save/Clear Queue Buttons: Interactive if queue has tasks and not processing.
    queue_management_interactive = queue_has_tasks and not is_processing

    return (
        gr.update(interactive=add_task_interactive, variant=add_task_variant),
        gr.update(interactive=process_queue_interactive, variant=process_queue_variant),
        gr.update(interactive=abort_task_interactive),
        gr.update(interactive=image_buttons_interactive),
        gr.update(interactive=image_buttons_interactive),
        gr.update(interactive=queue_management_interactive),
        gr.update(interactive=queue_management_interactive)
    )