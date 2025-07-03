# ui/queue.py
# Handles all user-facing queue management logic and event handling for the UI.

import gradio as gr
import numpy as np
from PIL import Image
import os
import json
import io
import zipfile # Keep this for save
import tempfile # Keep this for save
import logging

from .queue_manager import queue_manager_instance
from . import shared_state as shared_state_module
from .enums import ComponentKey as K
from . import workspace as workspace_manager
from . import queue_helpers, agents

logger = logging.getLogger(__name__)

AUTOSAVE_FILENAME = "goan_autosave_queue.zip"

# This mapping must match the header order in layout.py
ACTION_COLUMN_MAP = {
    0: 'move_up',
    1: 'move_down',
    2: 'pause',
    3: 'edit',
    4: 'cancel'
}

def add_or_update_task_in_queue(*args_from_ui_controls_tuple):
    """
    Adds a new task to the queue or updates an existing one if in edit mode.
    """
    # The first argument is always the input image PIL object.
    input_image_pil = args_from_ui_controls_tuple[0]
    
    if not input_image_pil:
        gr.Warning("Input image is required!")
        # Return no-op updates for all expected outputs.
        return [gr.update()] * (len(shared_state_module.ALL_TASK_UI_KEYS) + 8)

    # The rest of the arguments are the UI control values.
    all_ui_values_tuple = args_from_ui_controls_tuple[1:]
    default_keys_map = workspace_manager.get_default_values_map()
    # Use ALL_TASK_UI_KEYS to ensure the order and completeness of parameters.
    params_from_ui = dict(zip(shared_state_module.ALL_TASK_UI_KEYS, all_ui_values_tuple))
    base_params_for_worker_dict = {
        worker_key: params_from_ui.get(ui_key) for ui_key, worker_key in shared_state_module.UI_TO_WORKER_PARAM_MAP.items()
    }
    img_np_data = np.array(input_image_pil) # type: ignore

    editing_task_id = queue_manager_instance.get_state().get("editing_task_id")
    if editing_task_id is not None:
        queue_manager_instance.update_task(editing_task_id, base_params_for_worker_dict, img_np_data)
        # After updating, exit edit mode to reset the UI.
        return cancel_edit_mode_action()
    else:
        queue_manager_instance.add_task(base_params_for_worker_dict, img_np_data)
        # After adding, just update the queue display and let other components be.
        # The switchboard expects a specific number of outputs.
        num_outputs = len(shared_state_module.ALL_TASK_UI_KEYS) + 8
        updates = [gr.update()] * num_outputs
        # Index 0 is APP_STATE, Index 1 is QUEUE_DF
        updates[1] = queue_helpers.update_queue_df_display()
        return updates

def cancel_edit_mode_action() -> dict:
    """Resets the UI to its default state and exits edit mode by returning a dict of updates."""
    queue_manager_instance.set_editing_task(None)
    
    updates = {
        K.QUEUE_DF: queue_helpers.update_queue_df_display(),
        K.INPUT_IMAGE_DISPLAY: gr.update(value=None, visible=False),
        K.IMAGE_FILE_INPUT: gr.update(visible=True, value=None),
        K.CLEAR_IMAGE_BUTTON: gr.update(interactive=False),
        K.DOWNLOAD_IMAGE_BUTTON: gr.update(interactive=False),
        K.ADD_TASK_BUTTON: gr.update(value="Add Task to Queue", variant="secondary"),
        K.CANCEL_EDIT_TASK_BUTTON: gr.update(visible=False)
    }

    # Add updates to reset all workspace controls to their defaults
    default_values = workspace_manager.get_default_values_map()
    for key, value in default_values.items():
        updates[key] = gr.update(value=value)
        
    return updates

def handle_queue_action_on_select(*args, evt: gr.SelectData) -> dict:
    """Handles queue actions and returns a dictionary of UI updates."""
    if evt.index is None:
        return {} # Return empty dict for no change

    row_index, col_index = evt.index
    action = ACTION_COLUMN_MAP.get(col_index)

    if not action:
        return {}

    queue_state = queue_manager_instance.get_state()
    queue = queue_state["queue"]
    task = queue[row_index] if 0 <= row_index < len(queue) else None

    if not task:
        return {}

    # --- Handle Delete Action ('cancel') ---
    if action == "cancel":
        queue_manager_instance.remove_task(row_index)
        # If we deleted the task we were editing, reset the whole UI
        if queue_state.get("editing_task_id") == task['id']:
            return cancel_edit_mode_action()
        else:
            # Otherwise, just update the queue display
            return {K.QUEUE_DF: queue_helpers.update_queue_df_display()}

    # --- Handle Edit Action ---
    elif action == "edit":
        task_to_edit = queue_manager_instance.get_task_to_edit(row_index)
        if not task_to_edit:
            return {}

        # Build a dictionary of updates for entering edit mode
        updates = cancel_edit_mode_action() # Start with a full UI reset
        updates[K.QUEUE_DF] = queue_helpers.update_queue_df_display() # Get current DF state
        updates[K.ADD_TASK_BUTTON] = gr.update(value="Update Task", variant="primary")
        updates[K.CANCEL_EDIT_TASK_BUTTON] = gr.update(visible=True)

        # Apply the specific task's parameters
        params_to_load = task_to_edit['params']
        for ui_key, worker_key in shared_state_module.UI_TO_WORKER_PARAM_MAP.items():
            if worker_key in params_to_load:
                updates[ui_key] = gr.update(value=params_to_load[worker_key])
        
        # Handle the image
        img_np = params_to_load.get('input_image')
        if isinstance(img_np, np.ndarray):
            updates[K.INPUT_IMAGE_DISPLAY] = gr.update(value=Image.fromarray(img_np), visible=True)
            updates[K.IMAGE_FILE_INPUT] = gr.update(visible=False)

        return updates
        
    # --- Handle Other Actions (move_up, move_down, pause) ---
    elif action in ['move_up', 'move_down']:
        queue_manager_instance.move_task(action, row_index)
        return {K.QUEUE_DF: queue_helpers.update_queue_df_display()}
    
    elif action == 'pause':
        agents.ProcessingAgent().send({"type": "pause"})
        # No immediate UI update needed; the agent will respond via its own queue.
        return {}
        
    return {}

def clear_task_queue_action():
    """Clears all pending tasks from the queue."""
    queue_manager_instance.clear_pending_tasks()
    return gr.update(), queue_helpers.update_queue_df_display()

def save_queue_to_zip():
    """Saves the current queue to a zip file."""
    logger.info("Attempting to save queue to zip...")
    queue = queue_manager_instance.get_state().get("queue")
    if not queue:
        gr.Info("Queue is empty. Nothing to save.")
        return [gr.update(), None]
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
            temp_zip_path = tmp_file.name
        with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            queue_manifest = []
            for task in queue:
                params_copy = task['params'].copy()
                input_image_np = params_copy.pop('input_image', None)
                manifest_entry = {"id": task['id'], "params": params_copy, "status": task.get("status", "pending")}
                if input_image_np is not None:
                    img_filename = f"task_{task['id']}_input.png"
                    manifest_entry['image_ref'] = img_filename
                    img = Image.fromarray(input_image_np)
                    with io.BytesIO() as buf:
                        img.save(buf, format='PNG')
                        zf.writestr(img_filename, buf.getvalue())
                queue_manifest.append(manifest_entry)
            zf.writestr(shared_state_module.QUEUE_STATE_JSON_IN_ZIP, json.dumps(queue_manifest, indent=4))
        gr.Info(f"Queue with {len(queue)} tasks prepared for download.")
        return [gr.update(), temp_zip_path]
    except Exception as e:
        gr.Warning("Failed to create queue zip file.")
        logger.error(f"Error saving queue to zip: {e}", exc_info=True)
        return [gr.update(), None]

def load_queue_from_zip(zip_file_or_path):
    """Loads a queue from a zip file, including task parameters and images."""
    filepath = None
    if isinstance(zip_file_or_path, str) and os.path.exists(zip_file_or_path):
        filepath = zip_file_or_path # type: ignore
    elif hasattr(zip_file_or_path, 'name') and zip_file_or_path.name and os.path.exists(zip_file_or_path.name):
        filepath = zip_file_or_path.name
    
    if not filepath:
        logger.info("No valid queue file found to load.")
        return gr.update(), gr.update()

    new_queue, next_id = queue_helpers.reconstruct_queue_from_zip(filepath)
    if new_queue:
        queue_manager_instance.load_queue(new_queue, next_id)
        gr.Info(f"Successfully loaded {len(new_queue)} tasks from {os.path.basename(filepath)}.")
    
    return gr.update(), queue_helpers.update_queue_df_display()
