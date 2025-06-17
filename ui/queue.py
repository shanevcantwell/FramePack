# ui/queue.py
import gradio as gr
import numpy as np
from PIL import Image
import os
import json
import base64
import io
import zipfile
import tempfile
import atexit
import traceback
from pathlib import Path
import time # CHANGED: Imported time module for multi-click abort logic

# Import shared state and constants from the dedicated module.
from . import shared_state
from generation_core import worker
from diffusers_helper.thread_utils import AsyncStream, async_run

# NEW: Import helper functions from queue_helpers.py
from .queue_helpers import ( # Removed duplicated functions
    np_to_base64_uri,
    get_queue_state,
    update_queue_df_display
)


# Configuration for the autosave feature.
AUTOSAVE_FILENAME = "goan_autosave_queue.zip"

# Removed duplicated functions (np_to_base64_uri, get_queue_state, update_queue_df_display)
# They are now imported from queue_helpers.py


def add_or_update_task_in_queue(state_dict_gr_state, *args_from_ui_controls_tuple):
    queue_state = get_queue_state(state_dict_gr_state) # Uses imported helper
    editing_task_id = queue_state.get("editing_task_id", None)
    input_images_pil_list = args_from_ui_controls_tuple[0]
    if isinstance(input_images_pil_list, Image.Image):
        input_images_pil_list = [input_images_pil_list]
    all_ui_values_tuple = args_from_ui_controls_tuple[1:]
    if not input_images_pil_list:
        gr.Warning("Input image is required!")
        return state_dict_gr_state, update_queue_df_display(queue_state), gr.update(value="Add Task to Queue" if editing_task_id is None else "Update Task"), gr.update(visible=editing_task_id is not None)
    temp_params_from_ui = dict(zip(shared_state.ALL_TASK_UI_KEYS, all_ui_values_tuple))
    base_params_for_worker_dict = {}
    for ui_key, worker_key in shared_state.UI_TO_WORKER_PARAM_MAP.items():
        if ui_key == 'gs_schedule_shape_ui':
            base_params_for_worker_dict[worker_key] = temp_params_from_ui.get(ui_key) != 'Off'
        else:
            base_params_for_worker_dict[worker_key] = temp_params_from_ui.get(ui_key)
    if editing_task_id is not None:
        if len(input_images_pil_list) > 1: gr.Warning("Cannot update task with multiple images. Cancel edit."); return state_dict_gr_state, update_queue_df_display(queue_state), gr.update(value="Update Task"), gr.update(visible=True)
        pil_img_for_update = input_images_pil_list[0][0] if isinstance(input_images_pil_list[0], tuple) else input_images_pil_list[0]
        if not isinstance(pil_img_for_update, Image.Image): gr.Warning("Invalid image format for update."); return state_dict_gr_state, update_queue_df_display(queue_state), gr.update(value="Update Task"), gr.update(visible=True)
        img_np_for_update = np.array(pil_img_for_update)
        with shared_state.queue_lock:
            task_found = False
            for task in queue_state["queue"]:
                if task["id"] == editing_task_id:
                    task["params"] = {**base_params_for_worker_dict, 'input_image': img_np_for_update}
                    task["status"] = "pending"
                    task_found = True
                    break
            if not task_found: gr.Warning(f"Task {editing_task_id} not found for update.")
            else: gr.Info(f"Task {editing_task_id} updated.")
            queue_state["editing_task_id"] = None
    else:
        tasks_added_count = 0; first_new_task_id = -1
        with shared_state.queue_lock:
            for img_obj in input_images_pil_list:
                pil_image = img_obj[0] if isinstance(img_obj, tuple) else img_obj
                if not isinstance(pil_image, Image.Image): gr.Warning("Skipping invalid image input."); continue
                img_np_data = np.array(pil_image)
                next_id = queue_state["next_id"]
                if first_new_task_id == -1: first_new_task_id = next_id
                task = {"id": next_id, "params": {**base_params_for_worker_dict, 'input_image': img_np_data}, "status": "pending"}
                queue_state["queue"].append(task); queue_state["next_id"] += 1; tasks_added_count += 1
        if tasks_added_count > 0: gr.Info(f"Added {tasks_added_count} task(s) (start ID: {first_new_task_id}).")
        else: gr.Warning("No valid tasks added.")
    return state_dict_gr_state, update_queue_df_display(queue_state), gr.update(value="Add Task(s) to Queue", variant="secondary"), gr.update(visible=False)

def cancel_edit_mode_action(state_dict_gr_state):
    queue_state = get_queue_state(state_dict_gr_state) # Uses imported helper
    if queue_state.get("editing_task_id") is not None: gr.Info("Edit cancelled."); queue_state["editing_task_id"] = None
    return state_dict_gr_state, update_queue_df_display(queue_state), gr.update(value="Add Task(s) to Queue", variant="secondary"), gr.update(visible=False)

def move_task_in_queue(state_dict_gr_state, direction: str, selected_indices_list: list):
    if not selected_indices_list or not selected_indices_list[0]: return state_dict_gr_state, update_queue_df_display(get_queue_state(state_dict_gr_state)) # Uses imported helper
    idx = int(selected_indices_list[0][0]); queue_state = get_queue_state(state_dict_gr_state); queue = queue_state["queue"] # Uses imported helper
    with shared_state.queue_lock:
        if direction == 'up' and idx > 0: queue[idx], queue[idx-1] = queue[idx-1], queue[idx]
        elif direction == 'down' and idx < len(queue) - 1: queue[idx], queue[idx+1] = queue[idx+1], queue[idx]
    return state_dict_gr_state, update_queue_df_display(queue_state) # Uses imported helper

def remove_task_from_queue(state_dict_gr_state, selected_indices_list: list):
    removed_task_id = None
    if not selected_indices_list or not selected_indices_list[0]: return state_dict_gr_state, update_queue_df_display(get_queue_state(state_dict_gr_state)), removed_task_id # Uses imported helper
    idx = int(selected_indices_list[0][0]); queue_state = get_queue_state(state_dict_gr_state); queue = queue_state["queue"] # Uses imported helper
    with shared_state.queue_lock:
        if 0 <= idx < len(queue): removed_task = queue.pop(idx); removed_task_id = removed_task['id']; gr.Info(f"Removed task {removed_task_id}.")
        else: gr.Warning("Invalid index for removal.")
    return state_dict_gr_state, update_queue_df_display(queue_state), removed_task_id # Uses imported helper

def handle_queue_action_on_select(evt: gr.SelectData, state_dict_gr_state, *ui_param_controls_tuple):
    if evt.index is None or evt.value not in ["↑", "↓", "✖", "✎"]:
        return [state_dict_gr_state, update_queue_df_display(get_queue_state(state_dict_gr_state))] + [gr.update()] * (len(shared_state.ALL_TASK_UI_KEYS) + 4) # Uses imported helper
    row_index, col_index = evt.index; button_clicked = evt.value; queue_state = get_queue_state(state_dict_gr_state); queue = queue_state["queue"]; processing = queue_state.get("processing", False) # Uses imported helper
    outputs_list = [state_dict_gr_state, update_queue_df_display(queue_state)] + [gr.update()] * (len(shared_state.ALL_TASK_UI_KEYS) + 4) # Uses imported helper
    if button_clicked == "↑":
        if processing and row_index == 0: gr.Warning("Cannot move processing task."); return outputs_list
        new_state, new_df = move_task_in_queue(state_dict_gr_state, 'up', [[row_index, col_index]]); outputs_list[0], outputs_list[1] = new_state, new_df
    elif button_clicked == "↓":
        if processing and row_index == 0: gr.Warning("Cannot move processing task."); return outputs_list
        if processing and row_index == 1: gr.Warning("Cannot move below processing task."); return outputs_list
        new_state, new_df = move_task_in_queue(state_dict_gr_state, 'down', [[row_index, col_index]]); outputs_list[0], outputs_list[1] = new_state, new_df
    elif button_clicked == "✖":
        if processing and row_index == 0: gr.Warning("Cannot remove processing task."); return outputs_list
        new_state, new_df, removed_id = remove_task_from_queue(state_dict_gr_state, [[row_index, col_index]]); outputs_list[0], outputs_list[1] = new_state, new_df
        if removed_id is not None and queue_state.get("editing_task_id", None) == removed_id:
            queue_state["editing_task_id"] = None
            outputs_list[2 + 1 + len(shared_state.ALL_TASK_UI_KEYS)] = gr.update(value="Add Task(s) to Queue", variant="secondary")
            outputs_list[2 + 1 + len(shared_state.ALL_TASK_UI_KEYS) + 1] = gr.update(visible=False)
    elif button_clicked == "✎":
        if processing and row_index == 0: gr.Warning("Cannot edit processing task."); return outputs_list
        if 0 <= row_index < len(queue):
            task_to_edit = queue[row_index]; task_id_to_edit = task_to_edit['id']; params_to_load_to_ui = task_to_edit['params']
            queue_state["editing_task_id"] = task_id_to_edit; gr.Info(f"Editing Task {task_id_to_edit}.")
            img_np_from_task = params_to_load_to_ui.get('input_image')
            outputs_list[2] = gr.update(value=[(Image.fromarray(img_np_from_task), "loaded_image")]) if isinstance(img_np_from_task, np.ndarray) else gr.update(value=None)
            for i, ui_key in enumerate(shared_state.ALL_TASK_UI_KEYS):
                worker_key = shared_state.UI_TO_WORKER_PARAM_MAP.get(ui_key)
                if worker_key in params_to_load_to_ui:
                    value_from_task = params_to_load_to_ui[worker_key]
                    outputs_list[3 + i] = gr.update(value="Linear" if value_from_task else "Off") if ui_key == 'gs_schedule_shape_ui' else gr.update(value=value_from_task)
            outputs_list[2 + len(shared_state.ALL_TASK_UI_KEYS) + 1] = gr.update(value="Update Task", variant="primary")
            outputs_list[2 + len(shared_state.ALL_TASK_UI_KEYS) + 2] = gr.update(visible=True)
        else: gr.Warning("Invalid index for edit.")
    return outputs_list

def clear_task_queue_action(state_dict_gr_state):
    queue_state = get_queue_state(state_dict_gr_state); # Uses imported helper
    queue = queue_state["queue"]; processing = queue_state["processing"]; cleared_count = 0
    with shared_state.queue_lock:
        if processing:
             if len(queue) > 1: cleared_count = len(queue) - 1; queue_state["queue"] = [queue[0]]; gr.Info(f"Cleared {cleared_count} pending tasks.")
             else: gr.Info("Only processing task in queue.")
        elif queue: cleared_count = len(queue); queue.clear(); gr.Info(f"Cleared {cleared_count} tasks.")
        else: gr.Info("Queue empty.")
    if not processing and cleared_count > 0 and os.path.isfile(AUTOSAVE_FILENAME):
         try: os.remove(AUTOSAVE_FILENAME); print(f"Cleared autosave: {AUTOSAVE_FILENAME}.")
         except OSError as e: print(f"Error deleting autosave: {e}")
    return state_dict_gr_state, update_queue_df_display(queue_state) # Uses imported helper

def save_queue_to_zip(state_dict_gr_state):
    queue_state = get_queue_state(state_dict_gr_state); # Uses imported helper
    queue = queue_state.get("queue", [])
    if not queue: gr.Info("Queue is empty. Nothing to save."); return state_dict_gr_state, ""
    zip_buffer = io.BytesIO(); saved_files_count = 0
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_manifest = []; image_paths_in_zip = {}
            for task in queue:
                params_copy = task['params'].copy(); task_id_s = task['id']; input_image_np_data = params_copy.pop('input_image', None)
                manifest_entry = {"id": task_id_s, "params": params_copy, "status": task.get("status", "pending")}
                if input_image_np_data is not None:
                    img_hash = hash(input_image_np_data.tobytes()); img_filename_in_zip = f"task_{task_id_s}_input.png"; manifest_entry['image_ref'] = img_filename_in_zip
                    if img_hash not in image_paths_in_zip:
                        img_save_path = os.path.join(tmpdir, img_filename_in_zip)
                        try: Image.fromarray(input_image_np_data).save(img_save_path, "PNG"); image_paths_in_zip[img_hash] = img_filename_in_zip; saved_files_count +=1
                        except Exception as e: print(f"Error saving image for task {task_id_s} in zip: {e}")
                queue_manifest.append(manifest_entry)
            manifest_path = os.path.join(tmpdir, "queue_manifest.json");
            with open(manifest_path, 'w', encoding='utf-8') as f: json.dump(queue_manifest, f, indent=4)
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.write(manifest_path, arcname="queue_manifest.json")
                for img_hash, img_filename_rel in image_paths_in_zip.items(): zf.write(os.path.join(tmpdir, img_filename_rel), arcname=img_filename_rel)
            zip_buffer.seek(0); zip_base64 = base64.b64encode(zip_buffer.getvalue()).decode('utf-8')
            gr.Info(f"Queue with {len(queue)} tasks ({saved_files_count} images) prepared for download.")
            return state_dict_gr_state, zip_base64
    except Exception as e: print(f"Error creating zip for queue: {e}"); traceback.print_exc(); gr.Warning("Failed to create zip data."); return state_dict_gr_state, ""
    finally: zip_buffer.close()

def load_queue_from_zip(state_dict_gr_state, uploaded_zip_file_obj):
    if not uploaded_zip_file_obj or not hasattr(uploaded_zip_file_obj, 'name') or not Path(uploaded_zip_file_obj.name).is_file(): gr.Warning("No valid file selected."); return state_dict_gr_state, update_queue_df_display(get_queue_state(state_dict_gr_state)) # Uses imported helper
    queue_state = get_queue_state(state_dict_gr_state); # Uses imported helper
    newly_loaded_queue = []; max_id_in_file = 0; loaded_image_count = 0; error_messages = []
    try:
        with tempfile.TemporaryDirectory() as tmpdir_extract:
            with zipfile.ZipFile(uploaded_zip_file_obj.name, 'r') as zf:
                if "queue_manifest.json" not in zf.namelist(): raise ValueError("queue_manifest.json not found in zip")
                zf.extractall(tmpdir_extract)
            manifest_path = os.path.join(tmpdir_extract, "queue_manifest.json")
            with open(manifest_path, 'r', encoding='utf-8') as f: loaded_manifest = json.load(f)
            for task_data in loaded_manifest:
                params_from_manifest = task_data.get('params', {}); task_id_loaded = task_data.get('id', 0); max_id_in_file = max(max_id_in_file, task_id_loaded)
                image_ref_from_manifest = task_data.get('image_ref'); input_image_np_data = None
                if image_ref_from_manifest:
                    img_path_in_extract = os.path.join(tmpdir_extract, image_ref_from_manifest)
                    if os.path.exists(img_path_in_extract):
                        try: input_image_np_data = np.array(Image.open(img_path_in_extract)); loaded_image_count +=1
                        except Exception as img_e: error_messages.append(f"Err loading img for task {task_id_loaded}: {img_e}")
                    else: error_messages.append(f"Missing img file for task {task_id_loaded}: {image_ref_from_manifest}")
                runtime_task = {"id": task_id_loaded, "params": {**params_from_manifest, 'input_image': input_image_np_data}, "status": "pending"}
                newly_loaded_queue.append(runtime_task)
        with shared_state.queue_lock: queue_state["queue"] = newly_loaded_queue; queue_state["next_id"] = max(max_id_in_file + 1, queue_state.get("next_id", 1))
        gr.Info(f"Loaded {len(newly_loaded_queue)} tasks ({loaded_image_count} images).")
        if error_messages: gr.Warning(" ".join(error_messages))
    except Exception as e: print(f"Error loading queue: {e}"); traceback.print_exc(); gr.Warning(f"Failed to load queue: {str(e)[:200]}")
    finally:
        if uploaded_zip_file_obj and hasattr(uploaded_zip_file_obj, 'name') and uploaded_zip_file_obj.name and tempfile.gettempdir() in os.path.abspath(uploaded_zip_file_obj.name):
            try: os.remove(uploaded_zip_file_obj.name)
            except OSError: pass
    return state_dict_gr_state, update_queue_df_display(queue_state) # Uses imported helper

def autosave_queue_on_exit_action(state_dict_gr_state_ref):
    print("Attempting to autosave queue on exit...")
    queue_state = get_queue_state(state_dict_gr_state_ref) # Uses imported helper
    if not queue_state.get("queue"): print("Autosave: Queue is empty."); return
    try:
        _dummy_state_ignored, zip_b64_for_save = save_queue_to_zip(state_dict_gr_state_ref)
        if zip_b64_for_save:
            with open(AUTOSAVE_FILENAME, "wb") as f: f.write(base64.b64decode(zip_b64_for_save))
            print(f"Autosave successful: Queue saved to {AUTOSAVE_FILENAME}")
        else: print("Autosave failed: Could not generate zip data.")
    except Exception as e: print(f"Error during autosave: {e}"); traceback.print_exc()

def autoload_queue_on_start_action(state_dict_gr_state):
    """
    Loads the queue from the autosave file if it exists.
    This function no longer sets UI states for buttons, as that is now handled
    exclusively by the process_task_queue_main_loop on startup.
    """
    queue_state = get_queue_state(state_dict_gr_state)
    df_update = update_queue_df_display(queue_state)
    if not queue_state["queue"] and Path(AUTOSAVE_FILENAME).is_file():
        print(f"Autoloading queue from {AUTOSAVE_FILENAME}...")
        class MockFilepath:
            def __init__(self, name): self.name = name
        temp_state_for_load = {"queue_state": queue_state.copy()}
        loaded_state_result, df_update_after_load = load_queue_from_zip(temp_state_for_load, MockFilepath(AUTOSAVE_FILENAME))
        if loaded_state_result["queue_state"]["queue"]:
            queue_state.update(loaded_state_result["queue_state"])
            df_update = df_update_after_load
            print(f"Autoload successful. Loaded {len(queue_state['queue'])} tasks.")
            try:
                os.remove(AUTOSAVE_FILENAME)
                print(f"Removed autosave file: {AUTOSAVE_FILENAME}")
            except OSError as e:
                print(f"Error removing autosave file '{AUTOSAVE_FILENAME}': {e}")
        else:
            print("Autoload: File existed but queue remains empty. Resetting queue.")
            queue_state["queue"] = []
            queue_state["next_id"] = 1
            df_update = update_queue_df_display(queue_state)
    
    # Only return the state and the dataframe.
    return state_dict_gr_state, df_update

def process_task_queue_main_loop(state_dict_gr_state):
    queue_state = get_queue_state(state_dict_gr_state) # Uses imported helper
    shared_state.interrupt_flag.clear()
    # Reset the multi-level abort state at the start of every run
    shared_state.abort_state.update({'level': 0, 'last_click_time': 0})
    output_stream_for_ui = state_dict_gr_state.get("active_output_stream_queue")
    if queue_state["processing"]:
        gr.Info("Queue processing is already active. Attempting to re-attach UI to live updates...")
        if output_stream_for_ui is None:
            gr.Warning("No active stream found in state. Queue processing may have been interrupted. Please clear queue or restart."); queue_state["processing"] = False
            yield (state_dict_gr_state, update_queue_df_display(queue_state), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=True)); return # Uses imported helper
        yield (
            state_dict_gr_state,
            update_queue_df_display(queue_state), # Uses imported helper
            gr.update(value=state_dict_gr_state.get("last_completed_video_path", None)),
            gr.update(visible=True, value=None),
            gr.update(value=f"Re-attaching to processing Task {queue_state['queue'][0]['id']}... Awaiting next preview."),
            gr.update(value="<div style='text-align: center;'>Re-connecting...</div>"),
            gr.update(interactive=False),
            gr.update(interactive=True),
            gr.update(interactive=True)
        )
    elif not queue_state["queue"]:
        gr.Info("Queue is empty. Add tasks to process.")
        yield (state_dict_gr_state, update_queue_df_display(queue_state), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=True)); return # Uses imported helper
    else:
        queue_state["processing"] = True
        output_stream_for_ui = AsyncStream()
        state_dict_gr_state["active_output_stream_queue"] = output_stream_for_ui
        yield (state_dict_gr_state, update_queue_df_display(queue_state), gr.update(), gr.update(visible=False), gr.update(value="Queue processing started..."), gr.update(value=""), gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=True)) # Uses imported helper
    actual_output_queue = output_stream_for_ui.output_queue if output_stream_for_ui else None
    if not actual_output_queue:
        gr.Warning("Internal error: Output queue not available. Aborting."); queue_state["processing"] = False; state_dict_gr_state["active_output_stream_queue"] = None
        yield (state_dict_gr_state, update_queue_df_display(queue_state), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=True)); return # Uses imported helper
    while queue_state["queue"] and not shared_state.interrupt_flag.is_set():
        with shared_state.queue_lock:
            if not queue_state["queue"]: break
            current_task_obj = queue_state["queue"][0]
            task_parameters_for_worker = current_task_obj["params"]
            current_task_id = current_task_obj["id"]
        if task_parameters_for_worker.get('input_image') is None:
            print(f"Skipping task {current_task_id}: Missing input image data.")
            gr.Warning(f"Task {current_task_id} skipped: Input image is missing.")
            with shared_state.queue_lock:
                current_task_obj["status"] = "error"; current_task_obj["error_message"] = "Missing Image"
            yield (state_dict_gr_state, update_queue_df_display(queue_state), gr.update(), gr.update(visible=False), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=True)); break # Uses imported helper
        if task_parameters_for_worker.get('seed') == -1: task_parameters_for_worker['seed'] = np.random.randint(0, 2**32 - 1)
        print(f"Starting task {current_task_id} (Prompt: {task_parameters_for_worker.get('prompt', '')[:30]}...).")
        current_task_obj["status"] = "processing"
        yield (state_dict_gr_state, update_queue_df_display(queue_state), gr.update(), gr.update(visible=False), gr.update(value=f"Processing Task {current_task_id}..."), gr.update(value=""), gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=True)) # Uses imported helper
        worker_args = {
            **task_parameters_for_worker,
            'task_id': current_task_id, 'output_queue_ref': actual_output_queue,
            **shared_state.models
        }
        async_run(worker, **worker_args)
        last_known_output_filename = state_dict_gr_state.get("last_completed_video_path", None)
        task_completed_successfully = False
        while True:
            flag, data_from_worker = actual_output_queue.next()
            if flag == 'progress':
                msg_task_id, preview_np_array, desc_str, html_str = data_from_worker
                if msg_task_id == current_task_id: yield (state_dict_gr_state, update_queue_df_display(queue_state), gr.update(value=last_known_output_filename), gr.update(value=preview_np_array, visible=True), desc_str, html_str, gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=True))
            elif flag == 'file':
                msg_task_id, segment_file_path, segment_info = data_from_worker
                if msg_task_id == current_task_id: last_known_output_filename = segment_file_path; gr.Info(f"Task {current_task_id}: {segment_info}")
                yield (state_dict_gr_state, update_queue_df_display(queue_state), gr.update(value=last_known_output_filename), gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=True)) # Uses imported helper
            elif flag == 'aborted': current_task_obj["status"] = "aborted"; task_completed_successfully = False; break
            elif flag == 'error': _, error_message_str = data_from_worker; gr.Warning(f"Task {current_task_id} Error: {error_message_str}"); current_task_obj["status"] = "error"; current_task_obj["error_message"] = str(error_message_str)[:100]; task_completed_successfully = False; break
            elif flag == 'end': _, success_bool, final_video_path = data_from_worker; task_completed_successfully = success_bool; last_known_output_filename = final_video_path if success_bool else last_known_output_filename; current_task_obj["status"] = "done" if success_bool else "error"; break
        with shared_state.queue_lock:
            if queue_state["queue"] and queue_state["queue"][0]["id"] == current_task_id: queue_state["queue"].pop(0)
        state_dict_gr_state["last_completed_video_path"] = last_known_output_filename if task_completed_successfully else None
        final_desc = f"Task {current_task_id} {'completed' if task_completed_successfully else 'finished with issues'}."
        yield (state_dict_gr_state, update_queue_df_display(queue_state), gr.update(value=state_dict_gr_state["last_completed_video_path"]), gr.update(visible=False), gr.update(value=final_desc), gr.update(value=""), gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=True)) # Uses imported helper
        if shared_state.interrupt_flag.is_set(): gr.Info("Queue processing halted by user."); break
    queue_state["processing"] = False; state_dict_gr_state["active_output_stream_queue"] = None
    final_status_msg = "All tasks processed." if not shared_state.interrupt_flag.is_set() else "Queue processing aborted."
    yield (state_dict_gr_state, update_queue_df_display(queue_state), gr.update(value=state_dict_gr_state["last_completed_video_path"]), gr.update(visible=False), gr.update(value=final_status_msg), gr.update(value=""), gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=True)) # Uses imported helper

# CHANGED: Implemented multi-click logic for graceful/hard aborts.
def abort_current_task_processing_action(state_dict_gr_state):
    """Sends the abort signal to the currently processing task."""
    queue_state = get_queue_state(state_dict_gr_state) # Uses imported helper
    if not queue_state.get("processing", False):
        gr.Info("Nothing is currently processing.")
        return state_dict_gr_state, gr.update(interactive=False)

    # Also set the simple event for any part of the code that still uses it.
    shared_state.interrupt_flag.set()

    # Multi-click logic
    current_time = time.time()
    time_since_last_click = current_time - shared_state.abort_state.get('last_click_time', 0)

    # Check for double-click (e.g., within 0.75 seconds)
    if time_since_last_click < 0.75:
        shared_state.abort_state['level'] = 2
        gr.Info("Hard abort signal sent! Halting all operations.")
    else:
        shared_state.abort_state['level'] = 1
        gr.Info("Graceful abort signal sent. Will save preview after current step.")
    
    shared_state.abort_state['last_click_time'] = current_time
    
    return state_dict_gr_state, gr.update(interactive=True)
