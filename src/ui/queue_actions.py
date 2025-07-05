# ui/queue_actions.py
# Handles all user-facing queue management logic and event handling for the UI.

import gradio as gr
import numpy as np
from PIL import Image
import os
import json
import io
import zipfile # Keep this for save
import tempfile # Keep this for save
import shutil
import logging
import queue # For queue.Empty exception

from .queue_manager import queue_manager_instance
from . import shared_state as shared_state_module
from .enums import ComponentKey as K
from . import workspace as workspace_manager
from . import queue_helpers
from .agents import ProcessingAgent

logger = logging.getLogger(__name__)

AUTOSAVE_FILENAME = "goan_autosave_queue.zip"

def autosave_queue_on_exit_action():
    """Saves the current queue to a fixed autosave file on exit."""
    logger.info("Autosaving queue on exit...")
    queue = queue_manager_instance.get_state().get("queue")
    if not queue:
        logger.info("Queue is empty, nothing to autosave.")
        return
    try:
        # Use a similar logic to save_queue_to_zip but to a fixed path.
        with zipfile.ZipFile(AUTOSAVE_FILENAME, 'w', zipfile.ZIP_DEFLATED) as zf:
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
        logger.info(f"Successfully autosaved queue with {len(queue)} tasks to {AUTOSAVE_FILENAME}.")
    except Exception as e:
        logger.error(f"Error during queue autosave: {e}", exc_info=True)

def add_or_update_task_in_queue(*args_from_ui_controls_tuple):
    """
    Adds a new task to the queue or updates an existing one if in edit mode.
    This function is now independent of the old gr.State object.
    """
    # The first argument is always the input image PIL object.
    input_image_pil = args_from_ui_controls_tuple[0]
    
    if not input_image_pil:
        gr.Warning("Input image is required!")
        # Return updates for the outputs defined in the switchboard for this event.
        # We only need to return gr.update() for each output to signify no change.
        return [gr.update()] * (len(shared_state_module.ALL_TASK_UI_KEYS) + 8)

    # The rest of the arguments are the UI control values.
    all_ui_values_tuple = args_from_ui_controls_tuple[1:]
    default_keys_map = workspace_manager.get_default_values_map()
    enum_keys = list(default_keys_map.keys())
    params_from_ui = dict(zip(enum_keys, all_ui_values_tuple))
    base_params_for_worker_dict = {
        worker_key: params_from_ui.get(ui_key) for ui_key, worker_key in shared_state_module.UI_TO_WORKER_PARAM_MAP.items()
    }
    img_np_data = np.array(input_image_pil) # type: ignore

    editing_task_id = queue_manager_instance.get_state().get("editing_task_id")
    if editing_task_id is not None:
        queue_manager_instance.update_task(editing_task_id, base_params_for_worker_dict, img_np_data)
        # After updating a task, exit edit mode, which correctly resets the UI.
        return cancel_edit_mode_action()
    else:
        queue_manager_instance.add_task(base_params_for_worker_dict, img_np_data)
        # When adding a new task, we only need to update the queue display.
        # All other UI components remain as they are.
        # The switchboard expects a specific number of outputs, so we provide gr.update() placeholders.
        num_outputs = len(shared_state_module.ALL_TASK_UI_KEYS) + 8
        updates = [gr.update()] * num_outputs
        updates[1] = queue_helpers.update_queue_df_display() # Index 1 is the queue dataframe
        return updates

def cancel_edit_mode_action(from_ui=True):
    """Resets the UI to its default state and exits edit mode."""
    queue_manager_instance.set_editing_task(None)
    default_values_map = workspace_manager.get_default_values_map()
    ui_updates = [gr.update(value=default_values_map.get(key)) for key in shared_state_module.ALL_TASK_UI_KEYS]
    
    # The number of outputs depends on where this function is called from.
    # The switchboard has different output lists for different events.
    if from_ui:
        # This is for the "Cancel Edit" button click
        num_outputs = len(shared_state_module.ALL_TASK_UI_KEYS) + 8
        updates = [gr.update()] * num_outputs
        updates[1] = queue_helpers.update_queue_df_display()
        updates[2] = gr.update(value=None, visible=False) # INPUT_IMAGE_DISPLAY
        updates[3] = gr.update(visible=True, value=None) # QUEUE_DF
        for i, key in enumerate(shared_state_module.ALL_TASK_UI_KEYS):
            updates[4 + i] = ui_updates[i]
        updates[-4] = gr.update(interactive=False, variant="secondary") # CLEAR_IMAGE_BUTTON
        updates[-3] = gr.update(interactive=False, variant="secondary") # DOWNLOAD_IMAGE_BUTTON
        updates[-2] = gr.update(value="Add Task to Queue", variant="secondary") # ADD_TASK_BUTTON
        updates[-1] = gr.update(visible=True) # CANCEL_EDIT_TASK_BUTTON
        return updates
    else:
        # This is for when called internally, like after updating a task.
        # It matches the output signature of the add/update task event.
        return cancel_edit_mode_action(from_ui=True)
    
    # [svc] Add this missing function and dictionary back into ui/queue_actions.py

# This mapping must match the header order in the DataFrame defined in layout.py
ACTION_COLUMN_MAP = {
    0: 'move_up',
    1: 'move_down',
    2: 'pause',
    3: 'edit',
    4: 'cancel'
}

def handle_queue_action_on_select(evt: gr.SelectData, *args):
    """
    Handles user clicks on action icons within the queue DataFrame.
    This function was restored from a previous version to fix queue interactivity.
    """
    # Define the expected number of outputs to match the switchboard's 'add_task_outputs' list
    num_outputs = len(shared_state_module.ALL_TASK_UI_KEYS) + 8
    if evt.index is None:
        return [gr.update()] * num_outputs

    row_index, col_index = evt.index

    if col_index not in ACTION_COLUMN_MAP:
        logger.debug(f"Click on non-action column ({col_index}), ignoring.")
        return [gr.update()] * num_outputs

    action = ACTION_COLUMN_MAP[col_index]
    queue_state = queue_manager_instance.get_state()
    queue = queue_state.get("queue", [])

    if not (0 <= row_index < len(queue)):
        logger.warning(f"Invalid row index {row_index} for queue action.")
        return [gr.update()] * num_outputs

    task = queue[row_index]
    task_id = task['id']
    status = task.get("status", "pending")
    is_processing = status == 'processing' or (queue_state.get("processing", False) and row_index == 0)
    is_pending = status == 'pending'

    # --- Backend Enforcement of Disabled State ---
    if action in ['move_up', 'move_down', 'edit'] and not is_pending:
        gr.Info(f"Cannot '{action}' a task that is not 'Pending'.")
        return [gr.update()] * num_outputs
    if action == 'pause' and not is_processing:
        gr.Info("Can only pause a task that is currently 'Processing'.")
        return [gr.update()] * num_outputs

    # --- Handle Action ---
    if action == "move_up":
        queue_manager_instance.move_task('up', row_index)
    elif action == "move_down":
        queue_manager_instance.move_task('down', row_index)
    elif action == "cancel":
        if is_processing:
            gr.Info(f"Stopping and removing currently processing task {queue[0]['id']}...")
            ProcessingAgent().send({"type": "stop"})
            # Return immediately; the agent will send UI updates when the task is stopped.
            return [gr.update()] * num_outputs
        else:
            removed_id = queue_manager_instance.remove_task(row_index)
            # If we deleted the task we were editing, cancel edit mode.
            if removed_id is not None and queue_state.get("editing_task_id") == removed_id:
                return cancel_edit_mode_action()
    elif action == "edit":
        task_to_edit = queue_manager_instance.get_task_to_edit(row_index)
        if not task_to_edit:
            return [gr.update()] * num_outputs

        params_to_load_to_ui = task_to_edit['params']
        img_np_from_task = params_to_load_to_ui.get('input_image')
        img_display_update = gr.update(value=Image.fromarray(img_np_from_task), visible=True) if isinstance(img_np_from_task, np.ndarray) else gr.update(value=None, visible=False)
        file_input_update = gr.update(visible=False)
        
        # This mapping is complex, as it matches the 'add_task_outputs' in the switchboard
        ui_updates = [gr.update(value=params_to_load_to_ui.get(shared_state_module.UI_TO_WORKER_PARAM_MAP.get(key), None)) for key in workspace_manager.get_default_values_map().keys()]
        
        final_updates = (
            [gr.update(), queue_helpers.update_queue_df_display(), img_display_update, file_input_update] +
            ui_updates +
            [gr.update(interactive=True), gr.update(interactive=True), gr.update(value="Update Task", variant="primary"), gr.update(visible=True)]
        )
        return final_updates
    elif action == "pause":
        gr.Info(f"Requesting pause for task {task_id}...")
        ProcessingAgent().send({"type": "pause"})
        # Return immediately; the agent will send UI updates.
        return [gr.update()] * num_outputs

    # Default case: just update the queue display if a move or simple delete happened.
    updates = [gr.update()] * num_outputs
    updates[1] = queue_helpers.update_queue_df_display()
    return updates