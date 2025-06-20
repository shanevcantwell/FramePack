# ui/queue.py
# Handles all user-facing queue management logic and event handling for the UI.

import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image
import os
import json
import base64
import io
import zipfile
import tempfile
import traceback
import time
import shutil

from . import shared_state
from generation_core import worker
from diffusers_helper.thread_utils import AsyncStream, async_run
from . import queue_helpers

# Configuration for the autosave feature.
AUTOSAVE_FILENAME = "goan_autosave_queue.zip"

def add_or_update_task_in_queue(state_dict_gr_state, *args_from_ui_controls_tuple):
    """
    Adds a new task to the queue or updates an existing one if in edit mode.
    Gathers all settings from the UI controls.
    """
    queue_state = queue_helpers.get_queue_state(state_dict_gr_state)
    editing_task_id = queue_state.get("editing_task_id", None)
    
    input_images_pil_list = args_from_ui_controls_tuple[0]
    if isinstance(input_images_pil_list, Image.Image):
        input_images_pil_list = [input_images_pil_list]
    
    all_ui_values_tuple = args_from_ui_controls_tuple[1:]
    
    if not input_images_pil_list:
        gr.Warning("Input image is required!")
        return state_dict_gr_state, queue_helpers.update_queue_df_display(queue_state), gr.update(value="Add Task to Queue" if editing_task_id is None else "Update Task"), gr.update(visible=editing_task_id is not None)

    temp_params_from_ui = dict(zip(shared_state.ALL_TASK_UI_KEYS, all_ui_values_tuple))
    base_params_for_worker_dict = {
        worker_key: (temp_params_from_ui.get(ui_key) != 'Off' if ui_key == 'gs_schedule_shape_ui' else temp_params_from_ui.get(ui_key))
        for ui_key, worker_key in shared_state.UI_TO_WORKER_PARAM_MAP.items()
    }

    if editing_task_id is not None:
        pil_img_for_update = input_images_pil_list[0][0] if isinstance(input_images_pil_list[0], tuple) else input_images_pil_list[0]
        img_np_for_update = np.array(pil_img_for_update)
        with shared_state.queue_lock:
            for task in queue_state["queue"]:
                if task["id"] == editing_task_id:
                    task["params"] = {**base_params_for_worker_dict, 'input_image': img_np_for_update}
                    task["status"] = "pending"
                    gr.Info(f"Task {editing_task_id} updated.")
                    break
            queue_state["editing_task_id"] = None
    else:
        with shared_state.queue_lock:
            for img_obj in input_images_pil_list:
                pil_image = img_obj[0] if isinstance(img_obj, tuple) else img_obj
                img_np_data = np.array(pil_image)
                next_id = queue_state["next_id"]
                task = {"id": next_id, "params": {**base_params_for_worker_dict, 'input_image': img_np_data}, "status": "pending"}
                queue_state["queue"].append(task)
                queue_state["next_id"] += 1
            gr.Info(f"Added {len(input_images_pil_list)} task(s) to the queue.")

    return state_dict_gr_state, queue_helpers.update_queue_df_display(queue_state), gr.update(value="Add Task(s) to Queue", variant="secondary"), gr.update(visible=False)

def cancel_edit_mode_action(state_dict_gr_state):
    """Resets the UI from task-editing mode back to normal."""
    queue_state = queue_helpers.get_queue_state(state_dict_gr_state)
    if queue_state.get("editing_task_id") is not None:
        gr.Info("Edit cancelled.")
        queue_state["editing_task_id"] = None
    return state_dict_gr_state, queue_helpers.update_queue_df_display(queue_state), gr.update(value="Add Task(s) to Queue", variant="secondary"), gr.update(visible=False)

def handle_queue_action_on_select(evt: gr.SelectData, state_dict_gr_state, *ui_param_controls_tuple):
    """
    Event handler for clicks on the action icons (↑, ↓, ✖, ✎) in the queue DataFrame.
    """
    if evt.index is None or evt.value not in ["↑", "↓", "✖", "✎"]:
        # Return a matching number of updates to prevent errors
        return [state_dict_gr_state, queue_helpers.update_queue_df_display(queue_helpers.get_queue_state(state_dict_gr_state))] + [gr.update()] * (1 + len(shared_state.ALL_TASK_UI_KEYS) + 4)
    
    row_index, _ = evt.index
    button_clicked = evt.value
    queue_state = queue_helpers.get_queue_state(state_dict_gr_state)
    queue = queue_state["queue"]
    is_processing = queue_state.get("processing", False)
    
    if button_clicked in ["↑", "↓", "✖", "✎"] and is_processing and row_index == 0:
        gr.Warning("Cannot modify a task that is currently processing.")
        return [state_dict_gr_state, queue_helpers.update_queue_df_display(queue_state)] + [gr.update()] * (1 + len(shared_state.ALL_TASK_UI_KEYS) + 4)

    if button_clicked == "↑":
        queue_helpers.move_task_in_queue(state_dict_gr_state, 'up', row_index)
    elif button_clicked == "↓":
        queue_helpers.move_task_in_queue(state_dict_gr_state, 'down', row_index)
    elif button_clicked == "✖":
        _, removed_id = queue_helpers.remove_task_from_queue(state_dict_gr_state, row_index)
        if removed_id is not None and queue_state.get("editing_task_id") == removed_id:
             # If the task being edited was deleted, cancel edit mode
            return cancel_edit_mode_action(state_dict_gr_state)
    elif button_clicked == "✎":
        task_to_edit = queue[row_index]
        task_id_to_edit = task_to_edit['id']
        params_to_load_to_ui = task_to_edit['params']
        queue_state["editing_task_id"] = task_id_to_edit
        gr.Info(f"Editing Task {task_id_to_edit}.")
        
        img_np_from_task = params_to_load_to_ui.get('input_image')
        img_update = gr.update(value=[(Image.fromarray(img_np_from_task), "loaded_image")]) if isinstance(img_np_from_task, np.ndarray) else gr.update(value=None)
        
        ui_updates = [gr.update(value=params_to_load_to_ui.get(shared_state.UI_TO_WORKER_PARAM_MAP.get(key), None)) for key in shared_state.ALL_TASK_UI_KEYS]

        return ([state_dict_gr_state, queue_helpers.update_queue_df_display(queue_state), img_update] + ui_updates + 
                [gr.update(), gr.update(), gr.update(value="Update Task", variant="primary"), gr.update(visible=True)])

    return [state_dict_gr_state, queue_helpers.update_queue_df_display(queue_state)] + [gr.update()] * (1 + len(shared_state.ALL_TASK_UI_KEYS) + 4)

def clear_task_queue_action(state_dict_gr_state):
    """Clears all pending (non-processing) tasks from the queue."""
    queue_state = queue_helpers.get_queue_state(state_dict_gr_state)
    queue = queue_state["queue"]
    is_processing = queue_state["processing"]

    with shared_state.queue_lock:
        if is_processing:
            pending_tasks = queue[1:]
            cleared_count = len(pending_tasks)
            queue_state["queue"] = [queue[0]] # Keep only the processing task
            if cleared_count > 0:
                gr.Info(f"Cleared {cleared_count} pending tasks.")
        else:
            cleared_count = len(queue)
            if cleared_count > 0:
                queue.clear()
                gr.Info(f"Cleared {cleared_count} tasks from the queue.")
            else:
                gr.Info("Queue is already empty.")

    return state_dict_gr_state, queue_helpers.update_queue_df_display(queue_state)

def save_queue_to_zip(state_dict_gr_state):
    """Saves the entire current queue to a downloadable .zip file."""
    queue_state = queue_helpers.get_queue_state(state_dict_gr_state)
    queue = queue_state.get("queue", [])
    if not queue:
        gr.Info("Queue is empty. Nothing to save.")
        return state_dict_gr_state, None

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

            zf.writestr(shared_state.QUEUE_STATE_JSON_IN_ZIP, json.dumps(queue_manifest, indent=4))
        
        gr.Info(f"Queue with {len(queue)} tasks prepared for download.")
        return state_dict_gr_state, temp_zip_path
    except Exception as e:
        gr.Warning("Failed to create queue zip file.")
        print(f"Error saving queue to zip: {e}"); traceback.print_exc()
        return state_dict_gr_state, None

def _load_queue_from_zip_internal(app_state, zip_filepath):
    """
    Internal logic to load tasks from a zip file. Accepts a string filepath.
    This is called by both the manual load button and the autoload on start.
    """
    queue_state = queue_helpers.get_queue_state(app_state)
    newly_loaded_queue = []
    max_id_in_file = 0

    try:
        with zipfile.ZipFile(zip_filepath, 'r') as zf:
            if shared_state.QUEUE_STATE_JSON_IN_ZIP not in zf.namelist():
                raise FileNotFoundError(f"'{shared_state.QUEUE_STATE_JSON_IN_ZIP}' not found in zip.")
            
            manifest_data = json.loads(zf.read(shared_state.QUEUE_STATE_JSON_IN_ZIP))
            for task_data in manifest_data:
                task_id = task_data.get('id', 0)
                max_id_in_file = max(max_id_in_file, task_id)
                params = task_data.get('params', {})
                
                img_ref = task_data.get('image_ref')
                input_image_np = None
                if img_ref and img_ref in zf.namelist():
                    with zf.open(img_ref) as img_file:
                        input_image_np = np.array(Image.open(img_file))

                params['input_image'] = input_image_np
                newly_loaded_queue.append({"id": task_id, "params": params, "status": "pending"})

        with shared_state.queue_lock:
            queue_state["queue"] = newly_loaded_queue
            queue_state["next_id"] = max(max_id_in_file + 1, queue_state.get("next_id", 1))
        gr.Info(f"Successfully loaded {len(newly_loaded_queue)} tasks from {os.path.basename(zip_filepath)}.")
    except Exception as e:
        gr.Warning(f"Failed to load queue: {e}")
        print(f"Error loading queue from zip: {e}"); traceback.print_exc()

# --- REFACTORED GRADIO EVENT HANDLER ---
def load_queue_from_zip(state_dict_gr_state, uploaded_zip_file_obj):
    """
    Gradio event handler for the 'Load Queue' button. Extracts the filepath
    and calls the internal processing function.
    """
    if not uploaded_zip_file_obj or not hasattr(uploaded_zip_file_obj, 'name'):
        gr.Warning("No valid file selected.")
        return state_dict_gr_state, queue_helpers.update_queue_df_display(queue_helpers.get_queue_state(state_dict_gr_state))
    
    # Call the internal function with the path from the Gradio file object
    _load_queue_from_zip_internal(state_dict_gr_state, uploaded_zip_file_obj.name)
    
    return state_dict_gr_state, queue_helpers.update_queue_df_display(queue_helpers.get_queue_state(state_dict_gr_state))

# --- REFACTORED AUTOLOAD FUNCTION ---
def autoload_queue_on_start_action(app_state: dict) -> tuple:
    """
    Loads the last saved queue from the autosave file on UI startup.
    """
    print(f"Attempting to autoload queue from '{AUTOSAVE_FILENAME}'...")
    if os.path.exists(AUTOSAVE_FILENAME):
        # Directly call the internal function with the known filepath
        _load_queue_from_zip_internal(app_state, AUTOSAVE_FILENAME)

    queue_state = queue_helpers.get_queue_state(app_state)
    df_update = queue_helpers.update_queue_df_display(queue_state)
    
    return app_state, df_update, gr.update(), gr.update(), gr.update()

# (The rest of queue.py remains the same as the version I provided last)
# add_or_update_task_in_queue, cancel_edit_mode_action, handle_queue_action_on_select,
# clear_task_queue_action, save_queue_to_zip, autosave_queue_on_exit_action,
# process_task_queue_main_loop, abort_current_task_processing_action
def autosave_queue_on_exit_action(state_dict_gr_state_ref):
    """Saves the current queue to a zip file on application exit."""
    print("Attempting to autosave queue on exit...")
    queue_state = queue_helpers.get_queue_state(state_dict_gr_state_ref)
    if not queue_state.get("queue"):
        if os.path.exists(AUTOSAVE_FILENAME):
            try:
                os.remove(AUTOSAVE_FILENAME)
                print(f"Removed old autosave file: {AUTOSAVE_FILENAME}")
            except OSError as e:
                print(f"Error deleting old autosave file: {e}")
        return

    try:
        _, temp_zip_path = save_queue_to_zip(state_dict_gr_state_ref)
        if temp_zip_path and os.path.exists(temp_zip_path):
            shutil.copy(temp_zip_path, AUTOSAVE_FILENAME)
            os.remove(temp_zip_path)
            print(f"Autosave successful: Queue saved to {AUTOSAVE_FILENAME}")
    except Exception as e:
        print(f"Error during autosave: {e}"); traceback.print_exc()

def autoload_queue_on_start_action(app_state: dict) -> tuple:
    """
    Loads the last saved queue from the autosave file on UI startup.
    """
    print(f"Attempting to autoload queue from '{AUTOSAVE_FILENAME}'...")
    if os.path.exists(AUTOSAVE_FILENAME):
        mock_file_obj = type('MockFile', (), {'name': AUTOSAVE_FILENAME})()
        load_queue_from_zip(app_state, mock_file_obj)

    queue_state = queue_helpers.get_queue_state(app_state)
    df_update = queue_helpers.update_queue_df_display(queue_state)
    
    return app_state, df_update, gr.update(), gr.update(), gr.update()

def process_task_queue_main_loop(state_dict_gr_state, *lora_control_values):
    """
    Main loop for processing tasks. It streams progress updates to the UI.
    """
    queue_state = queue_helpers.get_queue_state(state_dict_gr_state)
    shared_state.interrupt_flag.clear()
    shared_state.abort_state.update({'level': 0, 'last_click_time': 0})
    
    if queue_state["processing"]:
        gr.Info("Queue processing is already active.")
        return
    if not queue_state["queue"]:
        gr.Info("Queue is empty. Add tasks to process.")
        return

    queue_state["processing"] = True
    output_stream = AsyncStream()
    state_dict_gr_state["active_output_stream_queue"] = output_stream
    
    yield (state_dict_gr_state, queue_helpers.update_queue_df_display(queue_state), gr.update(), gr.update(visible=False), 
           gr.update(value="Queue processing started..."), gr.update(value=""), gr.update(interactive=False), 
           gr.update(interactive=True), gr.update(interactive=True))

    while queue_state["queue"] and not shared_state.interrupt_flag.is_set():
        with shared_state.queue_lock:
            current_task = queue_state["queue"][0]
        
        if current_task.get('params', {}).get('seed') == -1:
            current_task['params']['seed'] = np.random.randint(0, 2**32 - 1)
        
        queue_helpers.apply_loras_from_state(state_dict_gr_state, *lora_control_values)
        current_task["status"] = "processing"
        
        yield (state_dict_gr_state, queue_helpers.update_queue_df_display(queue_state), gr.update(), 
               gr.update(visible=True), gr.update(value=f"Processing Task {current_task['id']}..."), 
               gr.update(value=""), gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=True))
        
        worker_args = {**current_task["params"], 'task_id': current_task['id'], 'output_queue_ref': output_stream.output_queue, **shared_state.models}
        async_run(worker, **worker_args)
        
        task_completed_successfully = False
        last_video_path = state_dict_gr_state.get("last_completed_video_path")

        while True:
            flag, data = output_stream.next()
            if flag == 'progress':
                _, preview_np, desc, html = data
                yield (state_dict_gr_state, gr.update(), gr.update(), gr.update(value=preview_np), desc, html, gr.update(), gr.update(), gr.update())
            elif flag == 'file':
                _, file_path, info_str = data
                last_video_path = file_path
                gr.Info(f"Task {current_task['id']}: {info_str}")
            elif flag == 'aborted':
                current_task["status"] = "aborted"; break
            elif flag == 'error':
                current_task["status"] = "error"; current_task["error_message"] = str(data[1])[:100]; break
            elif flag == 'end':
                task_completed_successfully, final_path = data[1], data[2]
                current_task["status"] = "done" if task_completed_successfully else "error"
                if task_completed_successfully: last_video_path = final_path
                break
        
        state_dict_gr_state["last_completed_video_path"] = last_video_path if task_completed_successfully else None
        
        with shared_state.queue_lock:
            if current_task["status"] == "done" and queue_state["queue"][0]["id"] == current_task["id"]:
                queue_state["queue"].pop(0)
        
        yield (state_dict_gr_state, queue_helpers.update_queue_df_display(queue_state), gr.update(value=last_video_path), 
               gr.update(visible=False), gr.update(value=f"Task {current_task['id']} finished."), gr.update(value=""), 
               gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=True))

        if shared_state.interrupt_flag.is_set():
            gr.Info("Queue processing halted by user.")
            break
            
    queue_state["processing"] = False
    state_dict_gr_state["active_output_stream_queue"] = None
    final_status = "All tasks processed." if not shared_state.interrupt_flag.is_set() else "Queue processing aborted."
    
    yield (state_dict_gr_state, queue_helpers.update_queue_df_display(queue_state), 
           gr.update(value=state_dict_gr_state.get("last_completed_video_path")), gr.update(visible=False), 
           gr.update(value=final_status), gr.update(value=""), gr.update(interactive=True), 
           gr.update(interactive=False), gr.update(interactive=True))

def abort_current_task_processing_action(state_dict_gr_state):
    """Sends a signal to gracefully or forcefully abort the current task."""
    queue_state = queue_helpers.get_queue_state(state_dict_gr_state)
    if not queue_state.get("processing", False):
        gr.Info("Nothing is currently processing.")
        return state_dict_gr_state, gr.update(interactive=False)

    shared_state.interrupt_flag.set()
    current_time = time.time()
    
    if (current_time - shared_state.abort_state.get('last_click_time', 0)) < 0.75:
        shared_state.abort_state['level'] = 2
        gr.Info("Hard abort signal sent! Halting all operations.")
    else:
        shared_state.abort_state['level'] = 1
        gr.Info("Graceful abort signal sent. Will stop after current step.")
    
    shared_state.abort_state['last_click_time'] = current_time
    return state_dict_gr_state, gr.update(interactive=True)