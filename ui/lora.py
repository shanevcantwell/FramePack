# ui/lora.py (Corrected for single file upload)

import gradio as gr
import os
import shutil
import json

LORA_DIR = os.path.abspath("./loras")
os.makedirs(LORA_DIR, exist_ok=True)

def handle_lora_upload_and_update_ui(app_state, uploaded_file):
    """
    Processes a single uploaded LoRA file, updates the application state, and
    immediately returns the necessary Gradio UI updates to show the controls.
    """
    # Part 1: Process file and update state
    # CHANGED: The 'uploaded_file' parameter is now a single file object, not a list.
    if uploaded_file is not None:
        lora_state = app_state.setdefault('lora_state', {})
        loaded_loras = lora_state.setdefault('loaded_loras', {})
        
        # This UI currently only supports one LoRA at a time.
        # Clear any previously loaded LoRA info.
        loaded_loras.clear()
        
        # CHANGED: Use the file object directly, no more list indexing.
        lora_name = os.path.basename(uploaded_file.name)
        persistent_path = os.path.join(LORA_DIR, lora_name)
        shutil.move(uploaded_file.name, persistent_path)
        
        loaded_loras[lora_name] = { "path": persistent_path }
        gr.Info(f"Loaded '{lora_name}'. UI updated.")

    # Part 2: Generate UI updates (this part remains the same)
    lora_state = app_state.get('lora_state', {})
    loaded_loras = lora_state.get('loaded_loras', {})
    
    if not loaded_loras:
        # If no LoRAs are loaded, hide the slot.
        ui_updates = [gr.update(value="[]"), gr.update(visible=False), gr.update(), gr.update(), gr.update()]
    else:
        # A LoRA is loaded, so create updates to show and populate the slot.
        lora_name = list(loaded_loras.keys())[0]
        ui_updates = [
            gr.update(value=json.dumps([lora_name])), # lora_name_state
            gr.update(visible=True),                  # lora_row_0 (make the whole row visible)
            gr.update(value=lora_name),               # lora_name_0
            gr.update(value=1.0),                     # lora_weight_0
            # CHANGED: Updated default targets to include both text encoders, a more common use case.
            gr.update(value=["text_encoder", "text_encoder_2"]) # lora_targets_0
        ]

    # Return the updated state PLUS the list of UI updates
    return [app_state] + ui_updates