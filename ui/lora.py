# ui/lora.py (Simplified for new pattern)

import gradio as gr
import os
import shutil
import json

LORA_DIR = os.path.abspath("./loras")
os.makedirs(LORA_DIR, exist_ok=True)

def handle_lora_upload(app_state, uploaded_files):
    """
    Handles file uploads, updates the app_state, and returns a list of
    gr.update objects to populate the pre-defined UI slots.
    """
    if not uploaded_files:
        return [app_state] + [gr.update()] * (1 + 5 * 4) # app_state + name_state + 5 rows/slots

    lora_state = app_state.get('lora_state', {"loaded_loras": {}})
    loaded_loras = lora_state.get('loaded_loras', {})
    
    # Create a list of gr.update objects to return
    updates = [gr.update()] * (1 + 5 * 4) # name_state + 5 * (row, name, weight, targets)

    # Process newly uploaded files
    for file_obj in uploaded_files:
        if len(loaded_loras) >= 5:
            gr.Warning("Maximum of 5 LoRAs reached.")
            break
            
        lora_name = os.path.basename(file_obj.name)
        if lora_name in loaded_loras:
            gr.Info(f"LoRA '{lora_name}' is already loaded.")
            continue

        persistent_path = os.path.join(LORA_DIR, lora_name)
        shutil.move(file_obj.name, persistent_path)
        
        loaded_loras[lora_name] = {
            "path": persistent_path
        }

    app_state['lora_state']['loaded_loras'] = loaded_loras
    
    # Update the UI based on the current state
    lora_names = list(loaded_loras.keys())
    updates[0] = json.dumps(lora_names) # Update the hidden name state textbox

    for i in range(5):
        if i < len(lora_names):
            name = lora_names[i]
            # Make the slot visible and set its values
            updates[1 + i * 4] = gr.update(visible=True)
            updates[2 + i * 4] = gr.update(value=name)
            updates[3 + i * 4] = gr.update(value=1.0) # Default weight
            updates[4 + i * 4] = gr.update(value=["transformer"]) # Default target
        else:
            # Hide unused slots
            updates[1 + i * 4] = gr.update(visible=False)

    return [app_state] + updates