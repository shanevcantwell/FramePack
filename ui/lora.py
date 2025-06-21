# ui/lora.py
# Manages LoRA uploads and the manual application/reversion of LoRA weights.

import os
import torch
import gradio as gr
import shutil
from safetensors.torch import load_file

from .enums import ComponentKey as K
from . import shared_state

# Define the directory where LoRAs are stored.
LORA_DIR = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'loras')))
os.makedirs(LORA_DIR, exist_ok=True)

# A mapping from the UI target selection to the model key in shared_state and the LoRA key prefix.
# This allows the manager to know that "transformer" in the UI corresponds to the 'transformer' model
# and that its LoRA keys start with 'lora_unet'.
LORA_TARGET_MAP = {
    "transformer": {"model_key": "transformer", "lora_prefix": "lora_unet"},
    "text_encoder": {"model_key": "text_encoder", "lora_prefix": "lora_te1"},
    "text_encoder_2": {"model_key": "text_encoder_2", "lora_prefix": "lora_te2"},
}

class LoRAManager:
    """
    Handles the manual application and removal of LoRA weights from models.
    This class directly modifies the model state_dicts and is designed to be
    used in a try/finally block to ensure weights are always reverted.
    """
    def __init__(self):
        self._original_weights = {}
        self._applied_loras = set()
        print("LoRAManager initialized.")

    def apply_lora(self, lora_name, weight_slider_val, target_modules):
        """
        Applies a LoRA to the registered models.

        Args:
            lora_name (str): The filename of the LoRA (e.g., "my_lora.safetensors").
            weight_slider_val (float): The multiplier for the LoRA weights.
            target_modules (list): A list of strings indicating which models to patch
                                   (e.g., ["transformer", "text_encoder"]).
        """
        if not lora_name or not target_modules:
            print("LoRA Manager: No LoRA name or target modules provided. Skipping application.")
            return

        lora_path = os.path.join(LORA_DIR, lora_name)
        if not os.path.exists(lora_path):
            print(f"Error: LoRA file not found at {lora_path}")
            return

        print(f"Applying LoRA '{lora_name}' with weight {weight_slider_val} to {target_modules}")

        try:
            lora_tensors = load_file(lora_path, device="cpu")
        except Exception as e:
            print(f"Error: Failed to load LoRA file {lora_path}: {e}")
            return

        for target_key in target_modules:
            if target_key not in LORA_TARGET_MAP:
                print(f"Warning: Unknown LoRA target '{target_key}'. Skipping.")
                continue

            map_info = LORA_TARGET_MAP[target_key]
            model = shared_state.models.get(map_info["model_key"])
            lora_prefix = map_info["lora_prefix"]

            if model is None:
                print(f"Warning: Model '{map_info['model_key']}' not found in shared state. Skipping.")
                continue

            self._patch_model(model, lora_tensors, lora_prefix, weight_slider_val)

        self._applied_loras.add(lora_name)

    @torch.no_grad()
    def _patch_model(self, model, lora_tensors, lora_prefix, alpha):
        """Patches a single model with the corresponding LoRA tensors."""
        modified_keys_count = 0
        model_state_dict = model.state_dict()
        
        # This structure assumes LoRA keys are like:
        # lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight
        # We derive the original model's key from that.
        lora_layers = {}
        for key, tensor in lora_tensors.items():
            if not key.startswith(lora_prefix):
                continue
            
            base_key = key.replace(lora_prefix + '_', '').split('.lora_')[0] + ".weight"
            
            if base_key not in lora_layers:
                lora_layers[base_key] = {}
            
            if "lora_down" in key:
                lora_layers[base_key]["down"] = tensor
            elif "lora_up" in key:
                lora_layers[base_key]["up"] = tensor

        for layer_key, tensors in lora_layers.items():
            if "up" not in tensors or "down" not in tensors:
                continue

            if layer_key not in model_state_dict:
                continue

            original_weight_param = model_state_dict[layer_key]

            # Backup the original weight if we haven't already
            if layer_key not in self._original_weights:
                self._original_weights[layer_key] = original_weight_param.clone().cpu()

            # Calculate and apply the delta
            lora_down = tensors["down"].to(original_weight_param.device, dtype=torch.float32)
            lora_up = tensors["up"].to(original_weight_param.device, dtype=torch.float32)
            
            delta = lora_up @ lora_down
            
            updated_weight = original_weight_param + delta.to(original_weight_param.dtype) * alpha
            original_weight_param.copy_(updated_weight)
            modified_keys_count += 1

        if modified_keys_count > 0:
            print(f"Patched {modified_keys_count} layers in {model.__class__.__name__}.")

    @torch.no_grad()
    def revert_all_loras(self):
        """Restores all modified model weights from the backup."""
        if not self._original_weights:
            return

        print(f"Reverting weights for LoRAs: {self._applied_loras}")
        reverted_count = 0
        
        for model_info in LORA_TARGET_MAP.values():
            model = shared_state.models.get(model_info["model_key"])
            if model is None:
                continue

            model_state_dict = model.state_dict()
            for layer_key, original_weight_cpu in self._original_weights.items():
                if layer_key in model_state_dict:
                    target_param = model_state_dict[layer_key]
                    target_param.copy_(original_weight_cpu.to(target_param.device, dtype=target_param.dtype))
                    reverted_count += 1
        
        print(f"Reverted {reverted_count} total layers.")
        self._original_weights.clear()
        self._applied_loras.clear()

def handle_lora_upload_and_update_ui(app_state, uploaded_file):
    """
    Processes a single uploaded LoRA file, saves it, and updates the UI.
    """
    # If the clear button was used, uploaded_file will be None.
    if uploaded_file is None:
        # For now, we don't clear state on clear, just hide UI.
        # A more robust implementation might remove from app_state here.
        return app_state, "", gr.update(visible=False), "", 1.0, []

    # A file was uploaded.
    lora_name = os.path.basename(uploaded_file.name)
    persistent_path = os.path.join(LORA_DIR, lora_name)
    
    # Use shutil.move to transfer the temporary file to the persistent loras directory.
    shutil.move(uploaded_file.name, persistent_path)
    print(f"Saved LoRA file to: {persistent_path}")
    
    # This simple UI only tracks one LoRA at a time. Clear any old ones.
    app_state["lora_state"]["loaded_loras"].clear()
    app_state["lora_state"]["loaded_loras"][lora_name] = { "path": persistent_path }

    gr.Info(f"Loaded '{lora_name}'.")

    # This Textbox is a trick to pass the lora name state around.
    lora_name_state_update = lora_name

    return (
        app_state,
        lora_name_state_update,
        gr.update(visible=True),                  # LORA_ROW_0
        gr.update(value=lora_name),               # LORA_NAME_0
        gr.update(value=1.0),                     # LORA_WEIGHT_0
        gr.update(value=["transformer"]) # LORA_TARGETS_0
    )