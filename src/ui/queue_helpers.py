# ui/queue_helpers.py
# Contains helper functions for queue state management and UI data formatting.

import gradio as gr
import numpy as np
from PIL import Image
import base64
import io
import logging

from .shared_state import shared_state_instance

logger = logging.getLogger(__name__)

def get_queue_state(state_dict_gr_state):
    """
    Safely retrieves the queue state dictionary from the main application state.
    Initializes a default queue state if it doesn't exist.
    """
    if "queue_state" not in state_dict_gr_state:
        state_dict_gr_state["queue_state"] = {
            "queue": [],
            "next_id": 1,
            "processing": False,
            "editing_task_id": None
        }
    return state_dict_gr_state["queue_state"]

def np_to_base64_uri(np_array_or_tuple, format="png"):
    """Converts a NumPy array to a base64 data URI for embedding in HTML/Markdown."""
    if np_array_or_tuple is None:
        return None
    try:
        # Handle cases where the input might be a raw array or a tuple from other components
        if isinstance(np_array_or_tuple, tuple) and len(np_array_or_tuple) > 0 and isinstance(np_array_or_tuple[0], np.ndarray):
            np_array = np_array_or_tuple[0]
        elif isinstance(np_array_or_tuple, np.ndarray):
            np_array = np_array_or_tuple
        else:
            return None
        
        pil_image = Image.fromarray(np_array.astype(np.uint8))
        if format.lower() == "jpeg" and pil_image.mode == "RGBA":
            pil_image = pil_image.convert("RGB")
        
        buffer = io.BytesIO()
        pil_image.save(buffer, format=format.upper())
        img_bytes = buffer.getvalue()
        return f"data:image/{format.lower()};base64,{base64.b64encode(img_bytes).decode('utf-8')}"
    except Exception as e:
        logger.error(f"Error converting NumPy to base64: {e}", exc_info=True)
        return None

def update_queue_df_display(queue_state):
    """Formats the current queue state into a Gradio DataFrame update object for display."""
    queue = queue_state.get("queue", [])
    data = []
    processing = queue_state.get("processing", False)
    editing_task_id = queue_state.get("editing_task_id", None)
    
    for i, task in enumerate(queue):
        params = task['params']
        task_id = task['id']
        
        # Create a truncated prompt for display and a full-text tooltip
        prompt_display = (params['prompt'][:77] + '...') if len(params['prompt']) > 80 else params['prompt']
        prompt_title = params['prompt'].replace('"', '&quot;')
        prompt_cell = f'<span title="{prompt_title}">{prompt_display}</span>'
        
        # Create an image thumbnail for the DataFrame
        img_uri = np_to_base64_uri(params.get('input_image'), format="png")
        thumbnail_size = "50px"
        img_md = f'<img src="{img_uri}" alt="Input" style="max-width:{thumbnail_size}; max-height:{thumbnail_size}; display:block; margin:auto; object-fit:contain;" />' if img_uri else ""
        
        # Determine the display status based on the task's state
        is_processing_current_task = processing and i == 0
        is_editing_current_task = editing_task_id == task_id
        task_status_val = task.get("status", "pending")

        if is_processing_current_task: status_display = "⏳ Processing"
        elif is_editing_current_task: status_display = "✏️ Editing"
        elif task_status_val == "done": status_display = "✅ Done"
        elif task_status_val == "error": status_display = f"❌ Error: {task.get('error_message', 'Unknown')}"
        elif task_status_val == "aborted": status_display = "⏹️ Aborted"
        else: status_display = "⏸️ Pending"
            
        data.append([
            task_id, status_display, "✖", "✎", # These match the new header order
            prompt_cell, f"{params.get('total_second_length', 0):.1f}s", img_md, "↑", "↓" # Remaining columns
        ])
        
    return gr.update(value=data)

def move_task_in_queue(state_dict_gr_state, direction: str, task_index: int):
    """Moves a task at a given index up or down in the queue."""
    queue_state = get_queue_state(state_dict_gr_state)
    queue = queue_state["queue"]
    
    with shared_state_instance.queue_lock:
        if direction == 'up' and task_index > 0:
            queue[task_index], queue[task_index-1] = queue[task_index-1], queue[task_index]
        elif direction == 'down' and task_index < len(queue) - 1:
            queue[task_index], queue[task_index+1] = queue[task_index+1], queue[task_index]
            
    return state_dict_gr_state

def remove_task_from_queue(state_dict_gr_state, task_index: int):
    """Removes a task at a given index from the queue."""
    queue_state = get_queue_state(state_dict_gr_state)
    queue = queue_state["queue"]
    removed_task_id = None
    
    with shared_state_instance.queue_lock:
        if 0 <= task_index < len(queue):
            removed_task = queue.pop(task_index)
            removed_task_id = removed_task['id']
            gr.Info(f"Removed task {removed_task_id}.")
        else:
            gr.Warning("Invalid index for removal.")
            
    return state_dict_gr_state, removed_task_id