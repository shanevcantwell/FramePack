# ui/lora.py (REVISED)
# Manages LoRA uploads and the dynamic application/reversion of LoRA layers.

import os
import torch
import math
import torch.nn as nn
import gradio as gr
import shutil
from safetensors.torch import load_file

from .enums import ComponentKey as K
from . import shared_state

# Define the directory where LoRAs are stored.
LORA_DIR = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'loras')))
os.makedirs(LORA_DIR, exist_ok=True)

# A mapping from the UI target selection to the model key in shared_state and the LoRA key prefix.
LORA_TARGET_MAP = {
    "transformer": {"model_key": "transformer", "lora_prefix": "lora_unet"},
    "text_encoder": {"model_key": "text_encoder", "lora_prefix": "lora_te1"},
    "text_encoder_2": {"model_key": "text_encoder_2", "lora_prefix": "lora_te2"},
}


class LoRALinearLayer(nn.Module):
    """
    A replacement for nn.Linear that incorporates LoRA logic.
    This layer wraps the original linear layer and applies the LoRA delta calculation
    during the forward pass.
    """
    def __init__(self, original_layer: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features

        # --- Store the original layer for reversion and its forward pass ---
        self.original_layer = original_layer
        
        # --- Create new LoRA parameters ---
        # These will be trained in a real fine-tuning scenario
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        
        # --- Initialize LoRA weights as per standard practice ---
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.scale = alpha / rank if rank > 0 else 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The original layer's properties (device and dtype) are the source of truth.
        original_device = self.original_layer.weight.device
        original_dtype = self.original_layer.weight.dtype

        # Perform the original forward pass.
        original_output = self.original_layer(x.to(device=original_device, dtype=original_dtype))

        if self.scale > 0:
            # Perform LoRA calculations on the input tensor's device and dtype.
            lora_A_dev = self.lora_A.to(device=x.device, dtype=x.dtype)
            lora_B_dev = self.lora_B.to(device=x.device, dtype=x.dtype)

            # W_orig*x + scale * (B @ A @ x)
            lora_delta = lora_B_dev @ (lora_A_dev @ x.transpose(-2, -1)).transpose(-2, -1)
            
            # MODIFIED: Explicitly cast the scaled delta to the same device AND dtype
            # as the original_output before adding them. This prevents dtype mismatches
            # (e.g., float32 + bfloat16) from nullifying the LoRA's effect.
            scaled_delta = (lora_delta * self.scale).to(device=original_device, dtype=original_output.dtype)
            
            return original_output + scaled_delta
        
        return original_output


def _find_and_set_module(model, module_key, new_module):
    """
    Recursively finds the parent of a module and replaces the child
    module with a new one.
    """
    tokens = module_key.split('.')
    parent_key = '.'.join(tokens[:-1])
    child_key = tokens[-1]
    
    parent_module = model
    for part in parent_key.split('.'):
        # Handle cases where keys contain indices like 'blocks.0.attn'
        if part.isdigit():
            parent_module = parent_module[int(part)]
        else:
            parent_module = getattr(parent_module, part)
            
    setattr(parent_module, child_key, new_module)


class LoRAManager:
    """
    Handles the dynamic replacement of model layers to apply and revert LoRAs.
    """
    def __init__(self):
        # Store a mapping of { 'module_key': original_layer } to revert changes
        self._replaced_layers = {}
        self._applied_loras = set()
        print("LoRAManager initialized with dynamic layer replacement strategy.")

    def apply_lora(self, lora_name, weight_slider_val, target_modules):
        if not lora_name or not target_modules:
            print("LoRA Manager: No LoRA name or target modules provided. Skipping application.")
            return

        lora_path = os.path.join(LORA_DIR, lora_name)
        if not os.path.exists(lora_path):
            print(f"Error: LoRA file not found at {lora_path}")
            return

        print(f"Applying LoRA '{lora_name}' with weight {weight_slider_val} to {target_modules}")

        try:
            # Load LoRA weights to CPU first to avoid device mismatches
            lora_tensors = load_file(lora_path, device="cpu")
        except Exception as e:
            print(f"Error: Failed to load LoRA file {lora_path}: {e}")
            return

        # --- Find rank and alpha from LoRA tensors ---
        # Typically found in a 'lora_down' weight tensor shape
        rank = 4 
        alpha = float(weight_slider_val)
        for key, tensor in lora_tensors.items():
            if "lora_down" in key and len(tensor.shape) == 2:
                rank = tensor.shape[0]
                break
        print(f"Determined LoRA rank: {rank}, Alpha: {alpha}")

        for target_key in target_modules:
            map_info = LORA_TARGET_MAP.get(target_key)
            if not map_info:
                print(f"Warning: Unknown LoRA target '{target_key}'. Skipping.")
                continue

            model = shared_state.models.get(map_info["model_key"])
            lora_prefix = map_info["lora_prefix"]
            if model is None:
                print(f"Warning: Model for '{target_key}' not found. Skipping.")
                continue

            self._patch_model(model, lora_tensors, lora_prefix, rank, alpha)

        self._applied_loras.add(lora_name)

    def _patch_model(self, model, lora_tensors, lora_prefix, rank, alpha):
        """Replaces linear layers with LoRALinearLayer."""

        # --- START: ADD THIS NEW DIAGNOSTIC BLOCK ---
        print("-" * 60)
        print(f"DIAGNOSTIC: LoRA Layer Matching for prefix: '{lora_prefix}'")
        print(f"Searching for tensor keys in LoRA file that start with '{lora_prefix}'.")
        print("-" * 60)

        # 1. Print all keys from the LoRA file to see what's available.
        print("Keys found in LoRA file:")
        lora_keys_found = [k for k in lora_tensors.keys() if k.startswith(lora_prefix)]
        if not lora_keys_found:
            print("  - WARNING: No keys found with the expected prefix!")
            print("  - All available keys in file are:")
            for k in lora_tensors.keys():
                print(f"    - {k}")
        else:
            for k in lora_keys_found:
                print(f"  - {k}")

        # 2. Print all targetable Linear layer names from the model.
        print("\nTargetable nn.Linear layers in model:")
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                print(f"  - {name}")
        print("-" * 60)
        # --- END: ADD THIS NEW DIAGNOSTIC BLOCK ---

        modified_keys_count = 0
        
        # --- Group LoRA up/down weights by the layer they target ---
        lora_layers_map = {}
        for key, tensor in lora_tensors.items():
            if not key.startswith(lora_prefix):
                continue
            
            # Convert LoRA key to its corresponding module key in the base model
            module_key = key.replace(lora_prefix + '_', '').split('.lora_')[0]
            
            if module_key not in lora_layers_map:
                lora_layers_map[module_key] = {}
            
            if "lora_down" in key:
                lora_layers_map[module_key]["down"] = tensor
            elif "lora_up" in key:
                lora_layers_map[module_key]["up"] = tensor

        # --- Iterate through target layers and replace them ---
        for module_key, tensors in lora_layers_map.items():
            if "up" not in tensors or "down" not in tensors:
                continue

            try:
                # Find the original linear layer in the model
                original_layer = model.get_submodule(module_key)
                if not isinstance(original_layer, nn.Linear):
                    continue

                # Create our new LoRA-aware layer
                new_layer = LoRALinearLayer(original_layer, rank, alpha)
                
                # Load the weights from the file into our new layer
                new_layer.lora_A.data.copy_(tensors["down"])
                new_layer.lora_B.data.copy_(tensors["up"])
                
                # Backup the original layer for reversion
                self._replaced_layers[module_key] = original_layer
                
                # Replace the layer on the model
                _find_and_set_module(model, module_key, new_layer)
                
                modified_keys_count += 1
            except Exception as e:
                print(f"Failed to patch module {module_key}: {e}")

        if modified_keys_count > 0:
            print(f"Replaced {modified_keys_count} layers in {model.__class__.__name__} with LoRA layers.")

    def revert_all_loras(self):
        """Restores all replaced layers from the backup."""
        if not self._replaced_layers:
            return

        print(f"Reverting layers for LoRAs: {self._applied_loras}")
        
        for model_info in LORA_TARGET_MAP.values():
            model = shared_state.models.get(model_info["model_key"])
            if model is None: continue

            for module_key, original_layer in self._replaced_layers.items():
                try:
                    # Check if the module key belongs to the current model being reverted
                    current_layer = model.get_submodule(module_key)
                    if isinstance(current_layer, LoRALinearLayer):
                         _find_and_set_module(model, module_key, original_layer)
                except Exception:
                    # This can happen if module_key doesn't exist in this specific model
                    pass
        
        print(f"Reverted {len(self._replaced_layers)} total layers.")
        self._replaced_layers.clear()
        self._applied_loras.clear()


def handle_lora_upload_and_update_ui(app_state, uploaded_file):
    """
    (This function remains unchanged from your original file)
    Processes a single uploaded LoRA file, saves it, and updates the UI.
    """
    if uploaded_file is None:
        return app_state, "", gr.update(visible=False), "", 1.0, []

    lora_name = os.path.basename(uploaded_file.name)
    persistent_path = os.path.join(LORA_DIR, lora_name)
    
    shutil.move(uploaded_file.name, persistent_path)
    print(f"Saved LoRA file to: {persistent_path}")
    
    app_state.get("lora_state", {}).get("loaded_loras", {}).clear()
    app_state.setdefault("lora_state", {}).setdefault("loaded_loras", {})[lora_name] = { "path": persistent_path }

    gr.Info(f"Loaded '{lora_name}'.")

    lora_name_state_update = lora_name

    return (
        app_state,
        lora_name_state_update,
        gr.update(visible=True),
        gr.update(value=lora_name),
        gr.update(value=0.8), # Default weight
        gr.update(value=["transformer"])
    )