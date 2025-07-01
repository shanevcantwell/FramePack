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
import shutil
import logging
import queue # For queue.Empty exception

from .queue_manager import queue_manager_instance
from . import shared_state as shared_state_module
from .enums import ComponentKey as K
from . import workspace as workspace_manager
from . import queue_helpers
from .agents import ProcessingAgent, ui_update_queue

logger = logging.getLogger(__name__)

AUTOSAVE_FILENAME = "goan_autosave_queue.zip"

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


def process_task_queue_and_listen(*lora_control_values):
    """Starts the ProcessingAgent, listens for UI updates, and handles stop requests."""
    agent = ProcessingAgent()

    # If processing is already active, this button click is a "stop" request.
    if queue_manager_instance.get_state().get("processing", False):
        # Set the flag for immediate UI feedback via update_button_states
        shared_state_module.shared_state_instance.stop_requested_flag.set()
        agent.send({"type": "stop"})
        gr.Info("Stop requested. The current task will be stopped.")
        # Return minimal updates. The .then() call in the switchboard will call
        # update_button_states, which will see the flag and update the UI correctly.
        return [gr.update()] * 9

    # If not processing, this is a "start" request.
    agent.send({
        "type": "start",
        "lora_controls": lora_control_values
    })

    # The listener loop. It doesn't manage state, just streams updates from the agent.
    while True:
        try:
            # Block until an update is available from the agent's UI queue.
            flag, data = ui_update_queue.get(timeout=1.0)

            if flag == "processing_started":
                # This is the first signal from the agent that it has started.
                # Update the UI to the "processing" state.
                yield ( # The first output (APP_STATE) is gr.update() as we don't modify it here.
                    gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(value="Queue processing started..."), # Progress description
                    gr.update(value=None, visible=True), # Progress bar
                    gr.update(interactive=True, value="⏹️ Stop Processing", variant="stop"), # PROCESS_QUEUE_BUTTON
                    gr.update(interactive=True), # CREATE_PREVIEW_BUTTON
                    gr.update(interactive=False) # CLEAR_QUEUE_BUTTON_UI
                )
            elif flag == "progress":
                # Unpack data: task_id, preview_np, desc, html
                _, preview_np, desc, html = data # type: ignore
                yield (gr.update(), gr.update(), gr.update(), gr.update(value=preview_np), desc, html, gr.update(), gr.update(), gr.update())
            elif flag == "file":
                # Unpack data: task_id, new_video_path, _
                _, new_video_path, _ = data # type: ignore
                yield (gr.update(), gr.update(), gr.update(value=new_video_path), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
            elif flag == "task_starting":
                task = data # type: ignore
                yield (gr.update(), queue_helpers.update_queue_df_display(), gr.update(), gr.update(), f"Processing Task {task['id']}...", gr.update(), gr.update(), gr.update(), gr.update())
            elif flag == "task_finished":
                yield (gr.update(), queue_helpers.update_queue_df_display(), gr.update(), gr.update(), f"Task {data['id']} {data['status']}.", gr.update(), gr.update(), gr.update(), gr.update())
            elif flag == "info":
                gr.Info(data)
            elif flag == "queue_finished":
                # The agent has signaled the end of all processing.
                logger.info("UI listener received 'queue_finished' signal. Exiting loop.")
                break

        except queue.Empty:
            # If the queue is empty, we check if the agent is still processing.
            # If not, it means the process finished or was stopped without a final signal.
            if not queue_manager_instance.get_state().get("processing", False):
                logger.info("UI listener detected processing has stopped. Exiting loop.")
                break
            continue # Continue waiting for updates.

    # The .then() call in the switchboard will handle the final button state update.
    # We just need to yield one last time to ensure the final queue state is displayed.
    logger.info("UI listener loop finished. Yielding final queue display.")
    yield ( # Yield a final update to refresh the queue display.
        gr.update(),
        queue_helpers.update_queue_df_display(),
        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    )


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
        updates[2] = gr.update(value=None, visible=False) # INPUT_IMAGE_DISPLAY_UI
        updates[3] = gr.update(visible=True, value=None) # IMAGE_FILE_INPUT_UI
        for i, key in enumerate(shared_state_module.ALL_TASK_UI_KEYS):
            updates[4 + i] = ui_updates[i]
        updates[-4] = gr.update(interactive=False, variant="secondary") # CLEAR_IMAGE_BUTTON_UI
        updates[-3] = gr.update(interactive=False, variant="secondary") # DOWNLOAD_IMAGE_BUTTON_UI
        updates[-2] = gr.update(value="Add Task to Queue", variant="secondary") # ADD_TASK_BUTTON
        updates[-1] = gr.update(visible=True) # CANCEL_EDIT_TASK_BUTTON
        return updates
    else:
        # This is for when called internally, like after updating a task.
        # It matches the output signature of the add/update task event.
        return cancel_edit_mode_action(from_ui=True)



def handle_queue_action_on_select(*args, evt: gr.SelectData):
    # The full list of UI components is passed in *args, but we don't need them here.
    # We only need the event data.
    num_outputs = len(shared_state_module.ALL_TASK_UI_KEYS) + 8
    if evt.index is None or evt.value not in ["↑", "↓", "✖", "✎"]:
        return [gr.update()] * num_outputs

    row_index, _ = evt.index
    button_clicked = evt.value
    queue_state = queue_manager_instance.get_state()
    queue = queue_state["queue"]

    is_processing_task = queue_state.get("processing", False) and row_index == 0

    if is_processing_task and button_clicked in ["↑", "↓"]:
        gr.Warning("Cannot modify a task that is currently processing.")
        return [gr.update()] * num_outputs
    if button_clicked == "↑":
        queue_manager_instance.move_task('up', row_index)
    elif button_clicked == "↓":
        queue_manager_instance.move_task('down', row_index)
    elif button_clicked == "✖":
        if is_processing_task:
            gr.Info(f"Stopping and removing currently processing task {queue[0]['id']}...")
            ProcessingAgent().send({"type": "stop"})
            # The agent will handle the task status change. We just update the display.
        else:
            removed_id = queue_manager_instance.remove_task(row_index)
            if removed_id is not None and queue_state.get("editing_task_id") == removed_id:
                # If we deleted the task we were editing, cancel edit mode.
                return cancel_edit_mode_action(from_ui=True)
    elif button_clicked == "✎":
        task_to_edit = queue_manager_instance.get_task_to_edit(row_index)
        if not task_to_edit:
            return [gr.update()] * num_outputs

        params_to_load_to_ui = task_to_edit['params']
        img_np_from_task = params_to_load_to_ui.get('input_image')
        img_display_update = gr.update(value=Image.fromarray(img_np_from_task), visible=True) if isinstance(img_np_from_task, np.ndarray) else gr.update(value=None, visible=False) # type: ignore
        file_input_update = gr.update(visible=False) # Hide the uploader when showing an image
        ui_updates = [gr.update(value=params_to_load_to_ui.get(shared_state_module.UI_TO_WORKER_PARAM_MAP.get(key), None)) for key in shared_state_module.ALL_TASK_UI_KEYS]
    num_outputs = len(shared_state_module.ALL_TASK_UI_KEYS) + 8
    queue = queue_state.get("queue", [])
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
    filepath = None
    if isinstance(zip_file_or_path, str) and os.path.exists(zip_file_or_path):
        filepath = zip_file_or_path # type: ignore
    elif hasattr(zip_file_or_path, 'name') and zip_file_or_path.name and os.path.exists(zip_file_or_path.name):
        filepath = zip_file_or_path.name
    if not filepath:
        logger.info("No valid queue file found to load.")
        return [gr.update(), queue_helpers.update_queue_df_display()]
    newly_loaded_queue, max_id_in_file, loaded_image_count, error_messages = [], 0, 0, []
    try:
        with tempfile.TemporaryDirectory() as tmpdir_extract:
            with zipfile.ZipFile(filepath, 'r') as zf:
                if shared_state_module.QUEUE_STATE_JSON_IN_ZIP not in zf.namelist():
                    raise ValueError(f"'{shared_state_module.QUEUE_STATE_JSON_IN_ZIP}' not found in zip")
                zf.extractall(tmpdir_extract)
            manifest_path = os.path.join(tmpdir_extract, shared_state_module.QUEUE_STATE_JSON_IN_ZIP)
            with open(manifest_path, 'r', encoding='utf-8') as f:
                loaded_manifest = json.load(f)
            for task_data in loaded_manifest:
                params = task_data.get('params', {})
                task_id = task_data.get('id', 0)
                max_id_in_file = max(max_id_in_file, task_id)
                image_ref = task_data.get('image_ref')
                img_np = None
                if image_ref:
                    img_path_in_extract = os.path.join(tmpdir_extract, image_ref)
                    if os.path.exists(img_path_in_extract):
                        try:
                            img_np = np.array(Image.open(img_path_in_extract))
                            loaded_image_count += 1
                        except Exception as img_e:
                            error_messages.append(f"Err loading img for task {task_id}: {img_e}")
                    else:
                        error_messages.append(f"Missing img file for task {task_id}: {image_ref}")
                params['input_image'] = img_np
                newly_loaded_queue.append({"id": task_id, "params": params, "status": "pending"})
        
        queue_manager_instance.load_queue(new
        queue_helpers.update_queue_df_display(),
        gr.update(value=None, visible=False), # INPUT_IMAGE_DISPLAY_UI
        gr.update(visible=True, value=None), # IMAGE_FILE_INPUT_UI
        *ui_updates, # All other UI controls
        gr.update(interactive=False, variant="secondary"), # CLEAR_IMAGE_BUTTON_UI
        gr.update(interactive=False, variant="secondary"), # DOWNLOAD_IMAGE_BUTTON_UI
        gr.update(value="Add Task to Queue", variant="secondary"), # ADD_TASK_BUTTON
        gr.update(visible=False) # CANCEL_EDIT_TASK_BUTTON
    ]


def handle_queue_action_on_select(*args, evt: gr.SelectData):
    # The full list of UI components is passed in *args, but we don't need them here.
    # We only need the event data.
    if evt.index is None or evt.value not in ["↑", "↓", "✖", "✎"]:
        num_outputs = len(shared_state_module.ALL_TASK_UI_KEYS) + 8
        return [gr.update()] * num_outputs

    row_index, _ = evt.index
    button_clicked = evt.value
    queue_state = queue_manager_instance.get_state()
    queue = queue_state["queue"]

    is_processing_task = queue_state.get("processing", False) and row_index == 0

    if is_processing_task and button_clicked in ["↑", "↓"]:
        gr.Warning("Cannot modify a task that is currently processing.")
        num_outputs = len(shared_state_module.ALL_TASK_UI_KEYS) + 8
        return [gr.update()] * num_outputs
    if button_clicked == "↑":
        queue_manager_instance.move_task('up', row_index)
    elif button_clicked == "↓":
        queue_manager_instance.move_task('down', row_index)
    elif button_clicked == "✖":
        if is_processing_task:
            gr.Info(f"Stopping and removing currently processing task {queue[0]['id']}...")
            ProcessingAgent().send({"type": "stop"})
            # The agent will handle the task status change. We just update the display.
        else:
            removed_id = queue_manager_instance.remove_task(row_index)
            if removed_id is not None and queue_state.get("editing_task_id") == removed_id:
                # If we deleted the task we were editing, cancel edit mode.
                return cancel_edit_mode_action()
    elif button_clicked == "✎":
        task_to_edit = queue_manager_instance.get_task_to_edit(row_index)
        if not task_to_edit:
            num_outputs = len(shared_state_module.ALL_TASK_UI_KEYS) + 8
            return [gr.update()] * num_outputs

        params_to_load_to_ui = task_to_edit['params']
        img_np_from_task = params_to_load_to_ui.get('input_image')
        img_display_update = gr.update(value=Image.fromarray(img_np_from_task), visible=True) if isinstance(img_np_from_task, np.ndarray) else gr.update(value=None, visible=False) # type: ignore
        file_input_update = gr.update(visible=False) # Hide the uploader when showing an image
        ui_updates = [gr.update(value=params_to_load_to_ui.get(shared_state_module.UI_TO_WORKER_PARAM_MAP.get(key), None)) for key in shared_state_module.ALL_TASK_UI_KEYS]
        queue_manager_instance.load_queue(newly_loaded_queue, max_id_in_file + 1)
        gr.Info(f"Loaded {len(newly_loaded_queue)} tasks ({loaded_image_count} images).")
        if error_messages: gr.Warning(" ".join(error_messages))
    except Exception as e:
        gr.Warning(f"Failed to load queue from {os.path.basename(filepath)}: {e}")
        logger.error(f"Error loading queue: {e}", exc_info=True)
    return [
        gr.update(),
        queue_helpers.update_queue_df_display(),
        gr.update(value=None, visible=False),  # INPUT_IMAGE_DISPLAY_UI
        gr.update(visible=True, value=None),  # IMAGE_FILE_INPUT_UI
        *[gr.update() for _ in shared_state_module.ALL_TASK_UI_KEYS],  # All other UI controls
        gr.update(interactive=False, variant="secondary"),  # CLEAR_IMAGE_BUTTON_UI
        gr.update(interactive=False, variant="secondary"),  # DOWNLOAD_IMAGE_BUTTON_UI
        gr.update(value="Add Task to Queue", variant="secondary"),  # ADD_TASK_BUTTON
        gr.update(visible=False),  # CANCEL_EDIT_TASK_BUTTON
    ]