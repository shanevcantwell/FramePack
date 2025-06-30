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
import logging

from . import lora as lora_manager
from . import shared_state as shared_state_module
from .enums import ComponentKey as K
from . import workspace as workspace_manager
from core.generation_core import worker
from diffusers_helper.thread_utils import AsyncStream, async_run
from . import queue_helpers

# Initialize logger for this module
logger = logging.getLogger(__name__)

AUTOSAVE_FILENAME = "goan_autosave_queue.zip"


def worker_wrapper(output_queue_ref, **kwargs):
    """
    A wrapper that calls the real worker in a try-except block
    to catch and report any backend exceptions to the console.
    """
    try:
        worker(output_queue_ref=output_queue_ref, **kwargs)
    except Exception as e:
        tb_str = traceback.format_exc()
        logger.error(f"--- BACKEND WORKER CRASHED ---\n{tb_str}\n--------------------------", exc_info=True)
        output_queue_ref.push(('crash', tb_str))


def add_or_update_task_in_queue(state_dict_gr_state, *args_from_ui_controls_tuple):
    queue_state = queue_helpers.get_queue_state(state_dict_gr_state)
    editing_task_id = queue_state.get("editing_task_id", None)
    input_image_pil = args_from_ui_controls_tuple[0]
    if not input_image_pil:
        gr.Warning("Input image is required!")
        return queue_helpers.update_queue_df_display()
   
    all_ui_values_tuple = args_from_ui_controls_tuple[1:]
    # Use the workspace's default map as the single source of truth for UI keys.
    default_keys_map = workspace_manager.get_default_values_map()
    enum_keys = [K[key.upper()] for key in default_keys_map.keys()]
    temp_params_from_ui = dict(zip(enum_keys, all_ui_values_tuple))
    # This fixes a bug where editing a task with "Roll-off" would not restore the UI state correctly.
    base_params_for_worker_dict = {
        worker_key: temp_params_from_ui.get(ui_key) for ui_key, worker_key in shared_state_module.UI_TO_WORKER_PARAM_MAP.items()
    }
    img_np_data = np.array(input_image_pil)
    if editing_task_id is not None:
        # Prepare updates for all UI controls to reset them to defaults after adding/updating
        default_values_map = workspace_manager.get_default_values_map()
        ui_updates_to_reset = [gr.update(value=default_values_map.get(key)) for key in shared_state_module.ALL_TASK_UI_KEYS]
        img_display_update_to_reset = gr.update(value=None, visible=False)
        file_input_update_to_reset = gr.update(visible=True, value=None)
        clear_image_button_update_to_reset = gr.update(interactive=False, variant="secondary")
        download_image_button_update_to_reset = gr.update(interactive=False, variant="secondary")
        add_task_button_update_to_reset = gr.update(value="Add Task to Queue", variant="secondary")
        cancel_edit_button_update_to_reset = gr.update(visible=False)
        with shared_state_module.shared_state_instance.queue_lock:
            for task in queue_state["queue"]:
                if task["id"] == editing_task_id:
                    task["params"] = {**base_params_for_worker_dict, 'input_image': img_np_data}
                    task["status"] = "pending"
                    gr.Info(f"Task {editing_task_id} updated.")
                    break
            queue_state["editing_task_id"] = None
    else:
        queue_manager_instance.add_task(base_params_for_worker_dict, img_np_data)
    
    # After adding/updating, reset the UI to its default state
    return cancel_edit_mode_action()


def cancel_edit_mode_action():
    queue_manager_instance.set_editing_task(None)
    default_values_map = workspace_manager.get_default_values_map()
    ui_updates = [gr.update(value=default_values_map.get(key)) for key in shared_state_module.ALL_TASK_UI_KEYS]
    return (
        queue_helpers.update_queue_df_display(),
        gr.update(value=None, visible=False), # INPUT_IMAGE_DISPLAY_UI
        gr.update(visible=True, value=None), # IMAGE_FILE_INPUT_UI
        *ui_updates, # All other UI controls
        gr.update(interactive=False, variant="secondary"), # CLEAR_IMAGE_BUTTON_UI
        gr.update(interactive=False, variant="secondary"), # DOWNLOAD_IMAGE_BUTTON_UI
        gr.update(value="Add Task to Queue", variant="secondary"), # ADD_TASK_BUTTON
        gr.update(visible=False) # CANCEL_EDIT_TASK_BUTTON
    )


def handle_queue_action_on_select(evt: gr.SelectData):
    if evt.index is None or evt.value not in ["↑", "↓", "✖", "✎"]:
        return [queue_helpers.update_queue_df_display()] + [gr.update()] * (len(shared_state_module.ALL_TASK_UI_KEYS) + 6)

    row_index, _ = evt.index
    button_clicked = evt.value
    queue = queue_state["queue"]

    if queue_state.get("processing", False) and row_index == 0:
        if button_clicked == "✖":
            gr.Info(f"Stopping and removing currently processing task {queue[0]['id']}...")
            shared_state_module.shared_state_instance.interrupt_flag.set()
            shared_state_module.shared_state_instance.abort_state['level'] = 2
            # The task will be removed by process_task_queue_main_loop once it's aborted.
            return [state_dict_gr_state, queue_helpers.update_queue_df_display(queue_state)] + [gr.update()] * (1 + 1 + len(shared_state_module.ALL_TASK_UI_KEYS) + 4)
        elif button_clicked in ["↑", "↓"]: # Only prevent moving for processing task
            gr.Warning("Cannot modify a task that is currently processing.")
            return [state_dict_gr_state, queue_helpers.update_queue_df_display(queue_state)] + [gr.update()] * (1 + 1 + len(shared_state_module.ALL_TASK_UI_KEYS) + 4)
        # If button_clicked is "✎", we now allow it to fall through to the edit logic below
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
        img_display_update = gr.update(value=Image.fromarray(img_np_from_task), visible=True) if isinstance(img_np_from_task, np.ndarray) else gr.update(value=None, visible=False) # type: ignore
        file_input_update = gr.update(visible=False) # Hide the uploader when showing an image
        # Corrected: UI_TO_WORKER_PARAM_MAP keys are K enums, not their string values.
        ui_updates = [gr.update(value=params_to_load_to_ui.get(shared_state_module.UI_TO_WORKER_PARAM_MAP.get(key), None)) for key in shared_state_module.ALL_TASK_UI_KEYS]
        return [state_dict_gr_state, queue_helpers.update_queue_df_display(queue_state), img_display_update, file_input_update] + ui_updates + [gr.update(), gr.update(), gr.update(value="Update Task", variant="primary"), gr.update(visible=True)]
    return [state_dict_gr_state, queue_helpers.update_queue_df_display(queue_state)] + [gr.update()] * (1 + 1 + len(shared_state_module.ALL_TASK_UI_KEYS) + 4)


def clear_task_queue_action(state_dict_gr_state):
    queue_state = queue_helpers.get_queue_state(state_dict_gr_state)
    queue = queue_state["queue"]
    with shared_state_module.shared_state_instance.queue_lock:
        initial_count = len(queue)
        # Keep tasks that are NOT 'pending'. This includes 'processing', 'done', 'error', 'aborted'.
        # A task's status defaults to 'pending' if not explicitly set.
        queue_state["queue"] = [task for task in queue if task.get("status", "pending") != "pending"]
        cleared_count = initial_count - len(queue_state["queue"])
        if cleared_count > 0:
            gr.Info(f"Cleared {cleared_count} pending tasks.")
        else:
            gr.Info("No pending tasks to clear.")
    return state_dict_gr_state, queue_helpers.update_queue_df_display(queue_state)


def request_preview_generation_action(state_dict_gr_state):
    """
    Sends a graceful stop signal to the current worker to generate a preview video,
    but allows the main queue processing to continue to the next task.
    """
    queue_state = queue_helpers.get_queue_state(state_dict_gr_state)
    if not queue_state.get("processing", False):
        gr.Info("Nothing is currently processing.")
        return state_dict_gr_state
    
    # Set the dedicated preview request flag. The worker will check this after each segment.
    shared_state_module.shared_state_instance.preview_request_flag.set()
    gr.Info("Preview requested. A video of the current segment will be generated when it completes.")
    
    return state_dict_gr_state

def save_queue_to_zip(state_dict_gr_state):
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
            zf.writestr(shared_state_module.QUEUE_STATE_JSON_IN_ZIP, json.dumps(queue_manifest, indent=4))
        gr.Info(f"Queue with {len(queue)} tasks prepared for download.")
        return state_dict_gr_state, temp_zip_path
    except Exception as e:
        gr.Warning("Failed to create queue zip file.")
        logger.error(f"Error saving queue to zip: {e}", exc_info=True)
        return state_dict_gr_state, None


def load_queue_from_zip(state_dict_gr_state, zip_file_or_path):
    queue_state = queue_helpers.get_queue_state(state_dict_gr_state)
    filepath = None
    if isinstance(zip_file_or_path, str) and os.path.exists(zip_file_or_path):
        filepath = zip_file_or_path # type: ignore
    elif hasattr(zip_file_or_path, 'name') and zip_file_or_path.name and os.path.exists(zip_file_or_path.name):
        filepath = zip_file_or_path.name
    if not filepath:
        logger.info("No valid queue file found to load.")
        return state_dict_gr_state, queue_helpers.update_queue_df_display(queue_state)
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
        with shared_state_module.shared_state_instance.queue_lock:
            queue_state["queue"] = newly_loaded_queue
            queue_state["next_id"] = max(max_id_in_file + 1, queue_state.get("next_id", 1))
        gr.Info(f"Loaded {len(newly_loaded_queue)} tasks ({loaded_image_count} images).")
        if error_messages: gr.Warning(" ".join(error_messages))
    except Exception as e:
        gr.Warning(f"Failed to load queue from {os.path.basename(filepath)}: {e}")
        logger.error(f"Error loading queue: {e}", exc_info=True)
    return state_dict_gr_state, queue_helpers.update_queue_df_display(queue_state)


def autosave_queue_on_exit_action(state_dict_gr_state_ref):
    logger.info("Attempting to autosave queue on exit...")
    queue_state = queue_helpers.get_queue_state(state_dict_gr_state_ref)
    if not queue_state.get("queue"):
        if os.path.exists(AUTOSAVE_FILENAME):
            try:
                os.remove(AUTOSAVE_FILENAME)
                logger.info(f"Autosave: Removed old (now empty) autosave file: {AUTOSAVE_FILENAME}")
            except OSError as e:
                logger.error(f"Autosave: Error deleting old autosave file: {e}")
        return
    try:
        _, temp_zip_path = save_queue_to_zip(state_dict_gr_state_ref)
        if temp_zip_path and os.path.exists(temp_zip_path):
            shutil.copy(temp_zip_path, AUTOSAVE_FILENAME)
            os.remove(temp_zip_path)
            logger.info(f"Autosave successful: Queue saved to {AUTOSAVE_FILENAME}")
    except Exception as e:
        logger.error(f"Error during autosave: {e}", exc_info=True)

def process_task_queue_main_loop(state_dict_gr_state, *lora_control_values): # noqa: C901
    """Main loop for processing tasks. It streams progress updates to the UI."""
    queue_state = queue_helpers.get_queue_state(state_dict_gr_state)

    if queue_state.get("processing", False):
        gr.Info("Stop signal sent. Waiting for current step to finish...")
        shared_state_module.shared_state_instance.stop_requested_flag.set()
        shared_state_module.shared_state_instance.preview_request_flag.clear()
        shared_state_module.shared_state_instance.interrupt_flag.set()  # Signal a hard stop for the queue loop
        shared_state_module.shared_state_instance.abort_state['level'] = 2  # Signal a hard stop for the worker
        logger.info("Stop signal sent to worker. Interrupt Level: 2.")
        # We no longer yield here. The .then() call to update_button_states will now handle the UI feedback.
        return

    # --- START LOGIC ---
    shared_state_module.shared_state_instance.interrupt_flag.clear()
    shared_state_module.shared_state_instance.abort_state.update({"level": 0, "last_click_time": 0})

    if not queue_state["queue"]:
        gr.Info("Queue is empty. Add tasks to process.")
        yield (
            state_dict_gr_state,
            queue_helpers.update_queue_df_display(queue_state),
            gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(interactive=False, value="Queue Empty", variant="secondary"),
            gr.update(interactive=False),
            gr.update(interactive=True), # CLEAR_QUEUE_BUTTON_UI
        )
        return

    queue_state["processing"] = True
    output_stream = AsyncStream()
    state_dict_gr_state["active_output_stream_queue"] = output_stream

    has_pending_tasks_at_start = any(task.get("status", "pending") == "pending" for task in queue_state["queue"])

    yield (
           state_dict_gr_state,
        queue_helpers.update_queue_df_display(queue_state),
        gr.update(),
        gr.update(visible=False),
        gr.update(value="Queue processing started..."),
        gr.update(value=""),
        gr.update(interactive=True, value="⏹️ Stop Processing", variant="stop"), # PROCESS_QUEUE_BUTTON
        gr.update(interactive=False), # Always start disabled, enable on valid segments
        gr.update(interactive=has_pending_tasks_at_start),
    )

    lora_handler = lora_manager.LoRAManager()
    try:
        if lora_control_values and len(lora_control_values) >= 3:
            lora_name, lora_weight, lora_targets = lora_control_values
            lora_handler.apply_lora(lora_name, lora_weight, lora_targets)

        while queue_state["queue"] and not shared_state_module.shared_state_instance.interrupt_flag.is_set():
            with shared_state_module.shared_state_instance.queue_lock:
                current_task = queue_state["queue"][0]

            if current_task.get("params", {}).get("seed") == -1:
                current_task["params"]["seed"] = np.random.randint(0, 2**32 - 1)

            current_task["status"] = "processing"

            # Check for pending tasks *excluding* the one that is now processing.
            has_pending_tasks_during_run = any(task.get("status", "pending") == "pending" for task in queue_state["queue"][1:])

            yield (
                state_dict_gr_state,
                queue_helpers.update_queue_df_display(queue_state),
                gr.update(),
                gr.update(visible=True),
                gr.update(value=f"Processing Task {current_task['id']}..."),
                gr.update(value=""),
                gr.update(interactive=True, value="⏹️ Stop Processing", variant="stop"), # PROCESS_QUEUE_BUTTON
                gr.update(interactive=False), # Always start disabled, enable on valid segments
                gr.update(interactive=has_pending_tasks_during_run),
            )

            worker_args = {**current_task["params"], "task_id": current_task["id"], **shared_state_module.shared_state_instance.models}
            worker_args.pop('transformer', None) # The worker doesn't take the transformer as a direct kwarg.
            async_run(worker_wrapper, output_queue_ref=output_stream.output_queue, **worker_args)

            last_video_path_for_task = None
            task_completed_successfully = False

            while True:
                # Add an explicit interrupt check here to make the Stop button more responsive.
                if shared_state_module.shared_state_instance.interrupt_flag.is_set():
                    break

                flag, data = output_stream.output_queue.next()

                if flag == "progress":
                    task_id, preview_np, desc, html = data

                    # This yield only updates the progress display components.
                    # Button state is handled by event_handlers.update_button_states,
                    # which is called by the main UI thread after specific events.
                    yield (state_dict_gr_state, gr.update(), gr.update(), gr.update(value=preview_np), desc, html, gr.update(), gr.update(), gr.update())
                elif flag == "file":
                    _, new_video_path, _ = data
                    last_video_path_for_task = new_video_path

                    # --- FIX: Explicitly update button states when a file is generated ---
                    # A file being saved (especially a preview) is a key time to update button states,
                    # as the preview_request_flag has just been cleared in the backend.
                    create_preview_interactive = not shared_state_module.shared_state_instance.preview_request_flag.is_set()
                    create_preview_variant = "primary" if create_preview_interactive else "secondary"
                    
                    has_pending_tasks_during_run = any(task.get("status", "pending") == "pending" for task in queue_state["queue"][1:])
                    clear_queue_variant = "stop" if has_pending_tasks_during_run else "secondary"

                    yield (state_dict_gr_state, gr.update(), gr.update(value=new_video_path), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(interactive=create_preview_interactive, variant=create_preview_variant), gr.update(interactive=has_pending_tasks_during_run, variant=clear_queue_variant))
                elif flag == "crash":
                    task_completed_successfully = False
                    gr.Warning(f"Task {current_task['id']} failed! Check console for traceback.")
                    current_task["status"] = "error"
                    current_task["error_message"] = "Worker process crashed."
                    break
                elif flag == "end":
                    # The 'end' signal from the worker indicates the worker thread has finished.
                    # 'data' here is (task_id, success_status, final_output_filename_from_worker)
                    worker_task_id, worker_success_status, worker_final_output_filename = data                    
                    if worker_success_status:
                        current_task["status"] = "done"
                        if worker_final_output_filename:
                            current_task["final_output_filename"] = worker_final_output_filename
                    else:
                        # If worker_success_status is False, it means either a crash or an abort.
                        # We need to check current_task["status"] which would have been set by "crash" or "aborted" flags.
                        # If it wasn't set by those, it implies an unhandled worker exit.
                        if current_task["status"] not in ["error", "aborted"]:
                            current_task["status"] = "error" # Default to error if not explicitly aborted/crashed
                            current_task["error_message"] = "Worker exited unexpectedly."
                    break
                elif flag == "aborted": # Handle explicit abort signal from worker
                    task_completed_successfully = False # Task did not complete successfully
                    gr.Info(f"Task {current_task['id']} aborted by user.")
                    current_task["status"] = "aborted"
                    # No error_message for aborts, it's a user action
                    break

            with shared_state_module.shared_state_instance.queue_lock:
                if current_task["status"] in ["done", "error", "aborted"] and queue_state["queue"] and queue_state["queue"][0]["id"] == current_task["id"]:
                    queue_state["queue"].pop(0)

            final_video_for_display = state_dict_gr_state.get("last_completed_video_path")
            if current_task["status"] == "done" and current_task.get("final_output_filename"):
                final_video_for_display = current_task["final_output_filename"]
                state_dict_gr_state["last_completed_video_path"] = final_video_for_display
            elif last_video_path_for_task:
                final_video_for_display = last_video_path_for_task

            # After a task is done, re-check for any remaining pending tasks.
            has_pending_tasks_after_task = any(task.get("status", "pending") == "pending" for task in queue_state["queue"])
            queue_is_empty_after_task = not bool(queue_state["queue"])
            queue_has_tasks_at_end = not queue_is_empty_after_task
            
            # Determine status message for the just-finished/aborted task
            task_status_message = ""
            if current_task["status"] == "done":
                task_status_message = f"Task {current_task['id']} finished."
            elif current_task["status"] == "aborted":
                task_status_message = f"Task {current_task['id']} aborted by user."
            elif current_task["status"] == "error":
                task_status_message = f"Task {current_task['id']} failed!"

            # Determine PROCESS_QUEUE_BUTTON state
            process_queue_button_text = "▶️ Process Queue"
            process_queue_button_variant = "primary"
            process_queue_button_interactive = queue_has_tasks_at_end # Can restart if tasks remain

            # Determine CREATE_PREVIEW_BUTTON state
            create_preview_interactive = not shared_state_module.shared_state_instance.preview_request_flag.is_set() and not queue_is_empty_after_task and not shared_state_module.shared_state_instance.interrupt_flag.is_set()
            create_preview_variant = "primary" if create_preview_interactive else "secondary"

            yield (
                state_dict_gr_state,
                queue_helpers.update_queue_df_display(queue_state),
                gr.update(value=final_video_for_display),
                gr.update(), # Keep the last preview visible
                gr.update(value=task_status_message),
                gr.update(value=""),
                gr.update(interactive=process_queue_button_interactive, value=process_queue_button_text, variant=process_queue_button_variant),
                gr.update(interactive=create_preview_interactive, variant=create_preview_variant),
                gr.update(interactive=has_pending_tasks_after_task),
            )

            if shared_state_module.shared_state_instance.interrupt_flag.is_set():
                gr.Info("Queue processing stopped by user.")
                break

            shared_state_module.shared_state_instance.abort_state["level"] = 0

    finally:
        logger.info("Processing finished. Reverting all LoRAs to clean up.")
        lora_handler.revert_all_loras()
        shared_state_module.shared_state_instance.stop_requested_flag.clear()

    queue_state["processing"] = False
    state_dict_gr_state["active_output_stream_queue"] = None
    final_status_message = "All tasks processed." if not shared_state_module.shared_state_instance.interrupt_flag.is_set() else "Queue processing stopped."
    final_video_to_show = state_dict_gr_state.get("last_completed_video_path") # This line was missing
    queue_has_tasks_at_end = bool(queue_state["queue"])

    yield (
        state_dict_gr_state,
        queue_helpers.update_queue_df_display(queue_state),
        gr.update(value=final_video_to_show),
        gr.update(visible=False),
        gr.update(value=final_status_message),
        gr.update(value=""),
        gr.update(interactive=queue_has_tasks_at_end, value="▶️ Process Queue", variant="primary"),
        gr.update(interactive=False),
        gr.update(interactive=queue_has_tasks_at_end),
    )