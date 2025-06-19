# ui/queue_helpers.py (Full updated file)
# Contains helper functions for queue state management and UI data formatting.

import gradio as gr
import numpy as np
from PIL import Image
import base64
import io

from . import shared_state

def get_queue_state(state_dict_gr_state):
    """Retrieves the queue state dictionary from the main application state."""
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
        print(f"Error converting NumPy to base64: {e}")
        return None

def update_queue_df_display(queue_state):
    """Formats the current queue state into a Gradio DataFrame update object."""
    queue = queue_state.get("queue", [])
    data = []
    processing = queue_state.get("processing", False)
    editing_task_id = queue_state.get("editing_task_id", None)
    
    for i, task in enumerate(queue):
        params = task['params']
        task_id = task['id']
        prompt_display = (params['prompt'][:77] + '...') if len(params['prompt']) > 80 else params['prompt']
        prompt_title = params['prompt'].replace('"', '&quot;')
        prompt_cell = f'<span title="{prompt_title}">{prompt_display}</span>'
        
        img_uri = np_to_base64_uri(params.get('input_image'), format="png")
        thumbnail_size = "50px"
        img_md = f'<img src="{img_uri}" alt="Input" style="max-width:{thumbnail_size}; max-height:{thumbnail_size}; display:block; margin:auto; object-fit:contain;" />' if img_uri else ""
        
        is_processing_current_task = processing and i == 0
        is_editing_current_task = editing_task_id == task_id
        task_status_val = task.get("status", "pending")

        if is_processing_current_task:
            status_display = "⏳ Processing"
        elif is_editing_current_task:
            status_display = "✏️ Editing"
        elif task_status_val == "done":
            status_display = "✅ Done"
        elif task_status_val == "error":
            status_display = f"❌ Error: {task.get('error_message', 'Unknown')}"
        elif task_status_val == "aborted":
            status_display = "⏹️ Aborted"
        else: # "pending"
            status_display = "⏸️ Pending"
            
        data.append([
            task_id,
            status_display,
            prompt_cell,
            f"{params.get('total_second_length', 0):.1f}s",
            params.get('steps', 0),
            img_md,
            "↑", "↓", "✖", "✎"
        ])
        
    return gr.update(value=data)

def move_task_in_queue(state_dict_gr_state, direction: str, task_index: int):
    """Moves a task at a given index up or down in the queue."""
    queue_state = get_queue_state(state_dict_gr_state)
    queue = queue_state["queue"]
    
    with shared_state.queue_lock:
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
    
    with shared_state.queue_lock:
        if 0 <= task_index < len(queue):
            removed_task = queue.pop(task_index)
            removed_task_id = removed_task['id']
            gr.Info(f"Removed task {removed_task_id}.")
        else:
            gr.Warning("Invalid index for removal.")
            
    return state_dict_gr_state, removed_task_id

# --- NEW FUNCTION ADDED HERE ---
def apply_loras_from_state(app_state, *lora_control_values):
    """
    Parses LoRA settings from the static UI slots and applies them to the models.
    """
    lora_state = app_state.get('lora_state', {})
    loaded_loras = lora_state.get('loaded_loras', {})

    # --- Update the main state with the live values from the UI ---
    value_iterator = iter(lora_control_values)
    # The values come in groups of 4: row, name, weight, targets
    for i in range(5): 
        try:
            _row_visibility = next(value_iterator) # We don't need the row visibility
            name = next(value_iterator)
            weight = next(value_iterator)
            targets = next(value_iterator)

            # If a name exists in this slot, update its data in the master state
            if name and name in loaded_loras:
                loaded_loras[name]['weight'] = weight
                loaded_loras[name]['target_models'] = targets
        except StopIteration:
            break # Stop if we run out of values for any reason

    # --- The rest of the function applies the now-updated state ---
    model_map = {
        "transformer": shared_state.models.get('transformer'),
        "text_encoder": shared_state.models.get('text_encoder'),
        "text_encoder_2": shared_state.models.get('text_encoder_2')
    }
    models_to_affect = [m for m in model_map.values() if hasattr(m, 'load_adapter')]

    # Unload previous adapters for a clean state
    for model in models_to_affect:
        if hasattr(model, 'get_adapter_names'):
            adapter_names = model.get_adapter_names()
            for name in adapter_names:
                if name != "default": model.delete_adapter(name)

    if not loaded_loras: 
        print("No active LoRAs to apply for this task.")
        return

    print("Applying LoRAs for the upcoming task...")

    # Load and set adapters based on the updated state
    for name, data in loaded_loras.items():
        weight, path = data.get("weight", 0), data.get("path")
        if weight == 0 or not path: continue

        print(f"  - Preparing '{name}' for targets: {data.get('target_models', [])} with weight {weight}")

        for target_name in data.get("target_models", []):
            model_obj = model_map.get(target_name)
            if model_obj:
                try:
                    # Load adapter with a unique name
                    model_obj.load_adapter(path, adapter_name=name)
                except Exception as e:
                    gr.Warning(f"Could not load LoRA '{name}' into {target_name}. Error: {e}")

    # This part of the logic may be simplified depending on the peft version,
    # as set_adapter might not be needed if weights are handled differently.
    # For now, we assume this pattern is for activating with weights.
    active_loras = {name: data['weight'] for name, data in loaded_loras.items() if data.get('weight', 0) != 0}
    if active_loras:
        for model in models_to_affect:
             if hasattr(model, "set_adapter"):
                # Activate all loaded LoRAs with their specified weights
                model.set_adapter(list(active_loras.keys()), list(active_loras.values()))
        gr.Info(f"Applied {len(active_loras)} LoRA(s).")