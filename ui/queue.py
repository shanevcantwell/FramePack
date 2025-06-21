# ui/queue.py
# Handles all user-facing queue management logic and event handling for the UI.

import gradio as gr
import numpy as np
from PIL import Image
import os
import json
import io
import zipfile
import tempfile
import traceback
import time
import shutil

from . import shared_state
from core.generation_core import worker
from diffusers_helper.thread_utils import AsyncStream, async_run
from . import queue_helpers

AUTOSAVE_FILENAME = "goan_autosave_queue.zip"

# --- ADDED: New wrapper function for robust error handling ---
def worker_wrapper(output_queue_ref, **kwargs):
    """
    A wrapper that calls the real worker in a try-except block
    to catch and report any backend exceptions to the console.
    """
    try:
        # The real worker is imported from core.generation_core
        worker(output_queue_ref=output_queue_ref, **kwargs)
    except Exception:
        # If the worker crashes, print the full traceback and send a 'crash' signal to the UI.
        tb_str = traceback.format_exc()
        print(f"--- BACKEND WORKER CRASHED ---\n{tb_str}\n--------------------------")
        output_queue_ref.push(('crash', tb_str))


def add_or_update_task_in_queue(state_dict_gr_state, *args_from_ui_controls_tuple):
    # ... (no changes in this function) ...
    queue_state = queue_helpers.get_queue_state(state_dict_gr_state)
    editing_task_id = queue_state.get("editing_task_id", None)
    input_image_pil = args_from_ui_controls_tuple[0]
    if not input_image_pil:
        gr.Warning("Input image is required!")
        return state_dict_gr_state, queue_helpers.update_queue_df_display(queue_state), gr.update(value="Add Task to Queue" if editing_task_id is None else "Update Task"), gr.update(visible=editing_task_id is not None)
    all_ui_values_tuple = args_from_ui_controls_tuple[1:]
    temp_params_from_ui = dict(zip(shared_state.ALL_TASK_UI_KEYS, all_ui_values_tuple))
    base_params_for_worker_dict = {
        worker_key: (temp_params_from_ui.get(ui_key) != 'Off' if ui_key == 'gs_schedule_shape_ui' else temp_params_from_ui.get(ui_key))
        for ui_key, worker_key in shared_state.UI_TO_WORKER_PARAM_MAP.items()
    }
    img_np_data = np.array(input_image_pil)
    if editing_task_id is not None:
        with shared_state.queue_lock:
            for task in queue_state["queue"]:
                if task["id"] == editing_task_id:
                    task["params"] = {**base_params_for_worker_dict, 'input_image': img_np_data}
                    task["status"] = "pending"
                    gr.Info(f"Task {editing_task_id} updated.")
                    break
            queue_state["editing_task_id"] = None
    else:
        with shared_state.queue_lock:
            next_id = queue_state["next_id"]
            task = {"id": next_id, "params": {**base_params_for_worker_dict, 'input_image': img_np_data}, "status": "pending"}
            queue_state["queue"].append(task)
            queue_state["next_id"] += 1
            gr.Info("Added 1 task to the queue.")
    return state_dict_gr_state, queue_helpers.update_queue_df_display(queue_state), gr.update(value="Add Task to Queue", variant="secondary"), gr.update(visible=False)

def cancel_edit_mode_action(state_dict_gr_state):
    # ... (no changes in this function) ...
    queue_state = queue_helpers.get_queue_state(state_dict_gr_state)
    if queue_state.get("editing_task_id") is not None:
        gr.Info("Edit cancelled.")
        queue_state["editing_task_id"] = None
    return state_dict_gr_state, queue_helpers.update_queue_df_display(queue_state), gr.update(value="Add Task to Queue", variant="secondary"), gr.update(visible=False)

def handle_queue_action_on_select(evt: gr.SelectData, state_dict_gr_state, *ui_param_controls_tuple):
    # ... (no changes in this function) ...
    if evt.index is None or evt.value not in ["↑", "↓", "✖", "✎"]:
        return [state_dict_gr_state, queue_helpers.update_queue_df_display(queue_helpers.get_queue_state(state_dict_gr_state))] + [gr.update()] * (1 + len(shared_state.ALL_TASK_UI_KEYS) + 4)
    row_index, _ = evt.index
    button_clicked = evt.value
    queue_state = queue_helpers.get_queue_state(state_dict_gr_state)
    queue = queue_state["queue"]
    if button_clicked in ["↑", "↓", "✖", "✎"] and queue_state.get("processing", False) and row_index == 0:
        gr.Warning("Cannot modify a task that is currently processing.")
        return [state_dict_gr_state, queue_helpers.update_queue_df_display(queue_state)] + [gr.update()] * (1 + len(shared_state.ALL_TASK_UI_KEYS) + 4)
    if button_clicked == "↑":
        queue_helpers.move_task_in_queue(state_dict_gr_state, 'up', row_index)
    elif button_clicked == "↓":
        queue_helpers.move_task_in_queue(state_dict_gr_state, 'down', row_index)
    elif button_clicked == "✖":
        _, removed_id = queue_helpers.remove_task_from_queue(state_dict_gr_state, row_index)
        if removed_id is not None and queue_state.get("editing_task_id") == removed_id:
            return cancel_edit_mode_action(state_dict_gr_state)
    elif button_clicked == "✎":
        task_to_edit = queue[row_index]
        params_to_load_to_ui = task_to_edit['params']
        queue_state["editing_task_id"] = task_to_edit['id']
        gr.Info(f"Editing Task {task_to_edit['id']}.")
        img_np_from_task = params_to_load_to_ui.get('input_image')
        img_update = gr.update(value=Image.fromarray(img_np_from_task)) if isinstance(img_np_from_task, np.ndarray) else gr.update(value=None)
        ui_updates = [gr.update(value=params_to_load_to_ui.get(shared_state.UI_TO_WORKER_PARAM_MAP.get(key), None)) for key in shared_state.ALL_TASK_UI_KEYS]
        return [state_dict_gr_state, queue_helpers.update_queue_df_display(queue_state), img_update] + ui_updates + [gr.update(), gr.update(), gr.update(value="Update Task", variant="primary"), gr.update(visible=True)]
    return [state_dict_gr_state, queue_helpers.update_queue_df_display(queue_state)] + [gr.update()] * (len(shared_state.ALL_TASK_UI_KEYS) + 5)

def clear_task_queue_action(state_dict_gr_state):
    # ... (no changes in this function) ...
    queue_state = queue_helpers.get_queue_state(state_dict_gr_state)
    queue = queue_state["queue"]
    with shared_state.queue_lock:
        if queue_state["processing"]:
            pending_tasks = queue[1:]
            cleared_count = len(pending_tasks)
            queue_state["queue"] = [queue[0]]
            if cleared_count > 0: gr.Info(f"Cleared {cleared_count} pending tasks.")
        else:
            cleared_count = len(queue)
            if cleared_count > 0:
                queue.clear()
                gr.Info(f"Cleared {cleared_count} tasks from the queue.")
            else:
                gr.Info("Queue is already empty.")
    return state_dict_gr_state, queue_helpers.update_queue_df_display(queue_state)

def save_queue_to_zip(state_dict_gr_state):
    # ... (no changes in this function) ...
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

def load_queue_from_zip(state_dict_gr_state, zip_file_or_path):
    # ... (no changes in this function) ...
    queue_state = queue_helpers.get_queue_state(state_dict_gr_state)
    filepath = None
    if isinstance(zip_file_or_path, str) and os.path.exists(zip_file_or_path):
        filepath = zip_file_or_path
    elif hasattr(zip_file_or_path, 'name') and zip_file_or_path.name and os.path.exists(zip_file_or_path.name):
        filepath = zip_file_or_path.name
    if not filepath:
        print("No valid queue file found to load.")
        return state_dict_gr_state, queue_helpers.update_queue_df_display(queue_state)
    newly_loaded_queue, max_id_in_file, loaded_image_count, error_messages = [], 0, 0, []
    try:
        with tempfile.TemporaryDirectory() as tmpdir_extract:
            with zipfile.ZipFile(filepath, 'r') as zf:
                if shared_state.QUEUE_STATE_JSON_IN_ZIP not in zf.namelist():
                    raise ValueError(f"'{shared_state.QUEUE_STATE_JSON_IN_ZIP}' not found in zip")
                zf.extractall(tmpdir_extract)
            manifest_path = os.path.join(tmpdir_extract, shared_state.QUEUE_STATE_JSON_IN_ZIP)
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
        with shared_state.queue_lock:
            queue_state["queue"] = newly_loaded_queue
            queue_state["next_id"] = max(max_id_in_file + 1, queue_state.get("next_id", 1))
        gr.Info(f"Loaded {len(newly_loaded_queue)} tasks ({loaded_image_count} images).")
        if error_messages: gr.Warning(" ".join(error_messages))
    except Exception as e:
        gr.Warning(f"Failed to load queue from {os.path.basename(filepath)}: {e}")
        print(f"Error loading queue: {e}"); traceback.print_exc()
    return state_dict_gr_state, queue_helpers.update_queue_df_display(queue_state)


def autosave_queue_on_exit_action(state_dict_gr_state_ref):
    # ... (no changes in this function) ...
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

# --- REWRITTEN FUNCTION ---
def process_task_queue_main_loop(state_dict_gr_state, *lora_control_values):
    """Main loop for processing tasks. It streams progress updates to the UI."""
    queue_state = queue_helpers.get_queue_state(state_dict_gr_state)
    shared_state.interrupt_flag.clear()
    shared_state.abort_state.update({'level': 0, 'last_click_time': 0})

    if queue_state.get("processing", False):
        gr.Info("Queue processing is already active.")
        yield (state_dict_gr_state, queue_helpers.update_queue_df_display(queue_state), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=True))
        return

    if not queue_state["queue"]:
        gr.Info("Queue is empty. Add tasks to process.")
        yield (state_dict_gr_state, queue_helpers.update_queue_df_display(queue_state), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=True))
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

        # MODIFIED: Call the new robust wrapper function instead of the raw worker
        worker_args = {**current_task["params"], 'task_id': current_task['id'], **shared_state.models}
        async_run(worker_wrapper, output_queue_ref=output_stream.output_queue, **worker_args)

        # This will hold the path to the latest video file generated by the current task
        last_video_path_for_task = None
        task_crashed = False

        # MODIFIED: Simplified message loop to align with demo script and handle crashes
        while True:
            flag, data = output_stream.output_queue.next()

            if flag == 'progress':
                _, preview_np, desc, html = data
                yield (state_dict_gr_state, gr.update(), gr.update(), gr.update(value=preview_np), desc, html, gr.update(), gr.update(), gr.update())
            
            elif flag == 'file':
                # Worker has saved a video file. Store its path, but don't update the UI yet.
                last_video_path_for_task = data[0] if isinstance(data, (list, tuple)) else data
            
            elif flag == 'crash':
                task_crashed = True
                gr.Warning(f"Task {current_task['id']} failed! Check console for traceback.")
                current_task["status"] = "error"
                current_task["error_message"] = "Worker process crashed."
                break

            elif flag == 'end':
                # Worker finished. The loop can end.
                if not task_crashed:
                    current_task["status"] = "done"
                break
        
        # --- This block runs AFTER the worker has finished or crashed ---
        
        # Update the persistent state ONLY if the task was a success and produced a video
        if not task_crashed and last_video_path_for_task:
            state_dict_gr_state["last_completed_video_path"] = last_video_path_for_task
        
        # Create the final UI update for the video player
        video_player_update = gr.update(value=last_video_path_for_task) if not task_crashed and last_video_path_for_task else gr.update()
        
        with shared_state.queue_lock:
            if current_task["status"] in ["done", "error", "aborted"] and queue_state["queue"] and queue_state["queue"][0]["id"] == current_task["id"]:
                queue_state["queue"].pop(0)
        
        yield (state_dict_gr_state, queue_helpers.update_queue_df_display(queue_state), video_player_update,
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
    # ... (no changes in this function) ...
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