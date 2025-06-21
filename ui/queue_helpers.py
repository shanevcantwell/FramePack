# ui/queue_helpers.py
# Contains helper functions for queue state management and UI data formatting.

import gradio as gr
import numpy as np
from PIL import Image
import base64
import io
from diffusers_helper.memory import DynamicSwapInstaller, cpu

from . import shared_state

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
        print(f"Error converting NumPy to base64: {e}")
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
            task_id, status_display, prompt_cell,
            f"{params.get('total_second_length', 0):.1f}s",
            params.get('steps', 0), img_md, "↑", "↓", "✖", "✎"
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

def apply_loras_from_state(app_state, *lora_control_values):
    """Parses LoRA settings from the UI and applies them to the models before a task runs."""
    lora_state = app_state.get('lora_state', {})
    loaded_loras_map = lora_state.get('loaded_loras', {})
    if not loaded_loras_map:
        return

    model_map = {
        "transformer": shared_state.models.get('transformer'),
        "text_encoder": shared_state.models.get('text_encoder'),
        "text_encoder_2": shared_state.models.get('text_encoder_2')
    }
    models_to_affect = [m for m in model_map.values() if m is not None and (hasattr(m, 'load_adapter') or 'DynamicSwap' in m.__class__.__name__)]

    # If DynamicSwap is active on the transformer, it must be temporarily removed
    # to allow the underlying model's adapter methods to be called directly.
    transformer_model = model_map.get("transformer")
    if transformer_model and 'DynamicSwap' in transformer_model.__class__.__name__:
        print("Temporarily uninstalling DynamicSwap wrapper to apply LoRA...")
        DynamicSwapInstaller.uninstall_model(transformer_model)

    # Unload all previous adapters to ensure a clean slate for the current task.
    for model in models_to_affect:
        if hasattr(model, 'get_adapter_names'):
            adapter_names = model.get_adapter_names()
            if "default" in adapter_names:
                adapter_names.remove("default")
            if adapter_names: model.delete_adapter(adapter_names)

    print("Applying LoRAs for the upcoming task...")
    active_loras_for_model = {}
    value_iterator = iter(lora_control_values)
    
    # This loop assumes a maximum number of LoRA slots from the UI.
    for _ in range(5): 
        try:
            name, weight, targets = next(value_iterator), next(value_iterator), next(value_iterator)
            if not (name and name in loaded_loras_map and weight != 0):
                continue
            
            lora_path = loaded_loras_map[name].get("path")
            if not lora_path: continue
            
            print(f"  - Loading '{name}' for targets: {targets}")
            active_loras_for_model[name] = weight

            for target_name in targets:
                model_obj = model_map.get(target_name)
                if model_obj:
                    try:
                        # Load the adapter by name, without applying weight here.
                        # Weights for all active adapters are applied in a batch later.
                        model_obj.load_adapter(lora_path, adapter_name=name)
                    except Exception as e:
                        gr.Warning(f"Could not load LoRA '{name}' into {target_name}. Error: {e}")
        except StopIteration:
            break
            
    # Apply weights to all loaded adapters in a single batch.
    # This is the correct method for multi-adapter setups with specified weights.
    if active_loras_for_model:
        adapter_names = list(active_loras_for_model.keys())
        adapter_weights = list(active_loras_for_model.values())
        for model in models_to_affect:
             if hasattr(model, "set_adapters"):
                model.set_adapters(adapter_names, adapter_weights)
        gr.Info(f"Applied {len(active_loras_for_model)} LoRA(s) with weights: {adapter_weights}")

    # Re-install the DynamicSwap wrapper if it was previously active.
    if transformer_model and 'DynamicSwap' not in transformer_model.__class__.__name__:
        print("Reinstalling DynamicSwap wrapper...")
        DynamicSwapInstaller.install_model(transformer_model, device=cpu)