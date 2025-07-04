# ui/event_handlers.py
import gradio as gr
import time
import tempfile
from PIL import Image, PngImagePlugin

import logging
from . import metadata as metadata_manager
from . import shared_state as shared_state_module
from . import workspace as workspace_manager
from . import queue as queue_actions # Use queue.py as the source for actions
from .enums import ComponentKey as K
from .queue_manager import queue_manager_instance
logger = logging.getLogger(__name__)


def reuse_last_seed_action(last_completed_seed):
    """Returns the last completed seed, or -1 if none exists."""
    if last_completed_seed is not None:
        gr.Info(f"Reusing last completed seed: {last_completed_seed}")
        return last_completed_seed
    else:
        gr.Warning("No task has been completed yet to reuse a seed from.")
        return gr.update() # Return no-op to not change the value

def safe_shutdown_action(app_state, *ui_values):
    """Performs all necessary save operations to prepare the app for a clean shutdown."""
    logger.info("Performing safe shutdown saves...")
    queue_actions.autosave_queue_on_exit_action()
    workspace_manager.save_ui_and_image_for_refresh(*ui_values)
    gr.Info("Queue and UI state saved. It is now safe to close the terminal.")

def ui_update_total_segments(total_seconds_ui, latent_window_size_ui, fps_ui):
    """Calculates and displays the number of segments based on video length."""
    try:
        logger.debug(f"ui_update_total_segments received: total_seconds_ui={total_seconds_ui}, latent_window_size_ui={latent_window_size_ui}, fps_ui={fps_ui}")
        total_frames = int(total_seconds_ui * fps_ui)
        frames_per_segment = latent_window_size_ui * 4 - 3
        total_segments = int(max(round(total_frames / frames_per_segment), 1)) if frames_per_segment > 0 else 1
        return f"Calculated: {total_segments} Segments, {total_frames} Total Frames"
    except (TypeError, ValueError):
        logger.error(f"Error in ui_update_total_segments. Inputs: total_seconds_ui={total_seconds_ui}, latent_window_size_ui={latent_window_size_ui}, fps_ui={fps_ui}", exc_info=True)
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
            gr.update(visible=True, value=None),    # IMAGE_FILE_INPUT
            gr.update(visible=False, value=None),   # INPUT_IMAGE_DISPLAY
            gr.update(interactive=False),           # CLEAR_IMAGE_BUTTON
            gr.update(interactive=False),           # DOWNLOAD_IMAGE_BUTTON
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
        gr.update(interactive=False, variant="secondary"), # CLEAR_IMAGE_BUTTON
        gr.update(interactive=False, variant="secondary"), # DOWNLOAD_IMAGE_BUTTON
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

# Define the button keys in a fixed order for consistent output.
BUTTON_KEYS = [
    K.ADD_TASK_BUTTON,
    K.PROCESS_QUEUE_BUTTON,
    K.CREATE_PREVIEW_BUTTON,
    K.CLEAR_IMAGE_BUTTON,
    K.DOWNLOAD_IMAGE_BUTTON,
    K.SAVE_QUEUE_BUTTON,
    K.CLEAR_QUEUE_BUTTON,
]

def get_button_state_outputs(components: dict) -> list:
    """Returns the list of button components for state updates."""
    return [components[key] for key in BUTTON_KEYS]

# --- MODIFICATION ---
# Removed the `queue_df_data` parameter from the function signature.
# This completes the fix by aligning the function definition with its call sites,
# which now correctly pass only two arguments. This resolves the UserWarning on startup.
def update_button_states(app_state, input_image_pil):
    """
    Updates button states based on a declarative rules engine. This function
    is the single source of truth for the state of all major control buttons.
    """
    # 1. Derive the current application state from the inputs.
    queue_state = queue_manager_instance.get_state()
    is_editing = queue_state.get("editing_task_id") is not None
    is_processing = queue_state.get("processing", False)
    is_editing_processing_task = is_editing and is_processing and queue_state.get("queue") and queue_state["editing_task_id"] == queue_state["queue"][0]["id"]
    state = {
        'stop_requested': shared_state_module.shared_state_instance.stop_requested_flag.is_set(),
        'is_editing': is_editing,
        'is_processing': is_processing,
        'is_editing_processing_task': is_editing_processing_task,
        'has_image': input_image_pil is not None,
        'queue_has_tasks': bool(queue_state.get("queue", [])),
        'has_pending_tasks': any(task.get("status", "pending") == "pending" for task in queue_state.get("queue", [])),
        'preview_requested': shared_state_module.shared_state_instance.preview_request_flag.is_set(),
    }

    # 2. Define the rules as a list of condition->updates mappings.
    # The first rule with a condition that returns True will be used.
    rules = [
        {'condition': lambda s: s['stop_requested'], 'get_updates': lambda s: {
            K.PROCESS_QUEUE_BUTTON: gr.update(interactive=False, value="Stopping...", variant="stop"),
            K.ADD_TASK_BUTTON: gr.update(interactive=False), K.CREATE_PREVIEW_BUTTON: gr.update(interactive=False),
            K.CLEAR_IMAGE_BUTTON: gr.update(interactive=False), K.DOWNLOAD_IMAGE_BUTTON: gr.update(interactive=False),
            K.SAVE_QUEUE_BUTTON: gr.update(interactive=False), K.CLEAR_QUEUE_BUTTON: gr.update(interactive=False),
        }},
        {'condition': lambda s: s['is_editing'], 'get_updates': lambda s: {
            K.ADD_TASK_BUTTON: gr.update(interactive=not s['is_editing_processing_task'], variant="primary"),
            K.PROCESS_QUEUE_BUTTON: gr.update(interactive=False, value="▶️ Process Queue", variant="secondary"),
            K.CREATE_PREVIEW_BUTTON: gr.update(interactive=False, variant="secondary"),
            K.CLEAR_IMAGE_BUTTON: gr.update(interactive=False, variant="secondary"), K.DOWNLOAD_IMAGE_BUTTON: gr.update(interactive=False, variant="secondary"),
            K.SAVE_QUEUE_BUTTON: gr.update(interactive=False, variant="secondary"), K.CLEAR_QUEUE_BUTTON: gr.update(interactive=False, variant="secondary"),
        }},
        {'condition': lambda s: s['is_processing'], 'get_updates': lambda s: {
            K.PROCESS_QUEUE_BUTTON: gr.update(interactive=True, value="⏹️ Stop Processing", variant="stop"),
            K.CREATE_PREVIEW_BUTTON: gr.update(interactive=not s['preview_requested'], variant="primary" if not s['preview_requested'] else "secondary"),
            K.CLEAR_QUEUE_BUTTON: gr.update(interactive=s['has_pending_tasks'], variant="stop" if s['has_pending_tasks'] else "secondary"),
            K.ADD_TASK_BUTTON: gr.update(interactive=False, variant="secondary"), K.CLEAR_IMAGE_BUTTON: gr.update(interactive=False, variant="secondary"),
            K.DOWNLOAD_IMAGE_BUTTON: gr.update(interactive=False, variant="secondary"), K.SAVE_QUEUE_BUTTON: gr.update(interactive=False, variant="secondary"),
        }},
        # Default rule for idle state.
        {'condition': lambda s: True, 'get_updates': lambda s: {
            K.ADD_TASK_BUTTON: gr.update(interactive=s['has_image'], variant="primary" if s['has_image'] else "secondary"),
            K.PROCESS_QUEUE_BUTTON: gr.update(interactive=s['queue_has_tasks'], value="▶️ Process Queue", variant="primary"),
            K.CREATE_PREVIEW_BUTTON: gr.update(interactive=False, variant="secondary"),
            K.CLEAR_IMAGE_BUTTON: gr.update(interactive=s['has_image'], variant="secondary"),
            K.DOWNLOAD_IMAGE_BUTTON: gr.update(interactive=s['has_image'], variant="secondary"),
            K.SAVE_QUEUE_BUTTON: gr.update(interactive=s['queue_has_tasks'], variant="primary"),
            K.CLEAR_QUEUE_BUTTON: gr.update(interactive=s['has_pending_tasks'], variant="stop" if s['has_pending_tasks'] else "secondary"),
        }},
    ]

    # 3. Find the first matching rule and get its updates dictionary.
    updates_dict = {}
    for rule in rules:
        if rule['condition'](state):
            updates_dict = rule['get_updates'](state)
            break

    # 4. Return the updates tuple in the correct, fixed order.
    return tuple(updates_dict.get(key, gr.update()) for key in BUTTON_KEYS)