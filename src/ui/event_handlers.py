# ui/event_handlers.py
import gradio as gr
import time
import tempfile
from PIL import Image, PngImagePlugin

from . import metadata as metadata_manager
from . import shared_state
from . import workspace as workspace_manager
# --- MODIFIED: Corrected the import paths for the queue functions ---
from .queue_helpers import get_queue_state
from .queue import autosave_queue_on_exit_action


def safe_shutdown_action(app_state, *ui_values):
    """Performs all necessary save operations to prepare the app for a clean shutdown."""
    print("Performing safe shutdown saves...")
    # This call to autosave is correct as it comes from queue.py
    autosave_queue_on_exit_action(app_state)
    workspace_manager.save_ui_and_image_for_refresh(*ui_values)
    gr.Info("Queue and UI state saved. It is now safe to close the terminal.")

def ui_update_total_segments(total_seconds_ui, latent_window_size_ui):
    """Calculates and displays the number of segments based on video length."""
    try:
        total_segments = int(max(round((total_seconds_ui * 30) / (latent_window_size_ui * 4)), 1))
        return f"Calculated Total Segments: {total_segments}"
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
        return (gr.update(visible=True, value=None), gr.update(visible=False, value=None), gr.update(visible=False),
                gr.update(visible=False), gr.update(variant="secondary"), "", {}, None)

    pil_image, prompt_preview, params = metadata_manager.open_and_check_metadata(filepath)

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
    is_processing = queue_state.get("processing", False)
    queue_has_tasks = bool(queue_state.get("queue"))

    add_task_variant = "primary" if input_image_pil is not None else "secondary"

    process_queue_interactive = not is_processing and queue_has_tasks
    process_queue_variant = "primary" if queue_has_tasks else "secondary"

    return (
        gr.update(variant=add_task_variant),
        gr.update(interactive=process_queue_interactive, variant=process_queue_variant)
    )