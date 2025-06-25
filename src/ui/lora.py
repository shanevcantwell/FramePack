# ui/lora.py (REVISED)
# REVISION 7: Restored the key translation logic that was missing from the
# previous version. This ensures LoRA keys are correctly matched to model layers.
# The comprehensive merge logic for Linear and Conv2d layers is retained.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
import shutil
import logging
from safetensors.torch import load_file
from core import model_loader # Import model_loader

from .enums import ComponentKey as K
from . import shared_state

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

LORA_DIR = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'loras')))
os.makedirs(LORA_DIR, exist_ok=True)

LORA_TARGET_MAP = {
    "transformer": {"model_key": "transformer", "lora_prefix": "lora_unet"},
    "text_encoder": {"model_key": "text_encoder", "lora_prefix": "lora_te1"},
    "text_encoder_2": {"model_key": "text_encoder_2", "lora_prefix": "lora_te2"},
}

def _get_module(model, module_key):
    """Helper function to retrieve a module from a model using its key."""
    parent_module = model
    for part in module_key.split('.'):
        parent_module = parent_module[int(part)] if part.isdigit() else getattr(parent_module, part)
    return parent_module

def _convert_hunyuan_keys_to_framepack(lora_sd: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    """Convert HunyuanVideo LoRA weights to FramePack format."""
    logger.info("Hunyuan LoRA detected, converting keys to FramePack format...")
    new_lora_sd = {}
    for key, weight in lora_sd.items():
        # Assumes keys are in kohya-ss format, e.g., lora_unet_...
        # We will replace parts of the key to match FramePack's layer names.
        new_key = key
        if "double_blocks" in new_key:
            new_key = new_key.replace("double_blocks", "transformer_blocks")
            new_key = new_key.replace("img_mod_linear", "norm1_linear")
            new_key = new_key.replace("img_attn_qkv", "attn_to_QKV")  # split later
            new_key = new_key.replace("img_attn_proj", "attn_to_out_0")
            new_key = new_key.replace("img_mlp_fc1", "ff_net_0_proj")
            new_key = new_key.replace("img_mlp_fc2", "ff_net_2")
            new_key = new_key.replace("txt_mod_linear", "norm1_context_linear")
            new_key = new_key.replace("txt_attn_qkv", "attn_add_QKV_proj")  # split later
            new_key = new_key.replace("txt_attn_proj", "attn_to_add_out")
            new_key = new_key.replace("txt_mlp_fc1", "ff_context_net_0_proj")
            new_key = new_key.replace("txt_mlp_fc2", "ff_context_net_2")
        elif "single_blocks" in new_key:
            new_key = new_key.replace("single_blocks", "single_transformer_blocks")
            new_key = new_key.replace("linear1", "attn_to_QKVM")  # split later
            new_key = new_key.replace("linear2", "proj_out")
            new_key = new_key.replace("modulation_linear", "norm_linear")
        else:
            new_lora_sd[key] = weight # Not a transformer block, keep as is.
            continue

        # --- Weight Splitting Logic ---
        if "QKVM" in new_key:
            # split QKVM into Q, K, V, M
            key_q = new_key.replace("QKVM", "q")
            key_k = new_key.replace("QKVM", "k")
            key_v = new_key.replace("QKVM", "v")
            key_m = new_key.replace("attn_to_QKVM", "proj_mlp")
            if "lora_down" in new_key or "alpha" in new_key:
                new_lora_sd[key_q] = weight
                new_lora_sd[key_k] = weight
                new_lora_sd[key_v] = weight
                new_lora_sd[key_m] = weight
            elif "lora_up" in new_key:
                # split QKVM weight into Q, K, V, M
                new_lora_sd[key_q] = weight[:3072]
                new_lora_sd[key_k] = weight[3072 : 3072 * 2]
                new_lora_sd[key_v] = weight[3072 * 2 : 3072 * 3]
                new_lora_sd[key_m] = weight[3072 * 3 :]
            else:
                logger.warning(f"Unsupported QKVM module name: {key}")
        elif "QKV" in new_key:
            # split QKV into Q, K, V
            key_q = new_key.replace("QKV", "q")
            key_k = new_key.replace("QKV", "k")
            key_v = new_key.replace("QKV", "v")
            if "lora_down" in new_key or "alpha" in new_key:
                new_lora_sd[key_q] = weight
                new_lora_sd[key_k] = weight
                new_lora_sd[key_v] = weight
            elif "lora_up" in new_key:
                # split QKV weight into Q, K, V
                new_lora_sd[key_q] = weight[:3072]
                new_lora_sd[key_k] = weight[3072 : 3072 * 2]
                new_lora_sd[key_v] = weight[3072 * 2 :]
            else:
                logger.warning(f"Unsupported QKV module name: {key}")
        else:
            # no split needed
            new_lora_sd[new_key] = weight

    return new_lora_sd

def _translate_lora_state_dict(lora_sd: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    """
    Checks if a LoRA state dict needs translation and applies it if necessary.
    """
    # Heuristic to detect if this is a Hunyuan-style LoRA that needs key conversion.
    is_hunyuan_format = any("double_blocks" in k or "single_blocks" in k for k in lora_sd.keys())
    if is_hunyuan_format:
        return _convert_hunyuan_keys_to_framepack(lora_sd, prefix)
    
    # If not a known incompatible format, return as is.
    return lora_sd

class LoRAManager:
    def __init__(self):
        # Store original params as {model_target_key: {module_path: param}}
        # e.g., {"transformer": {"block.0.attn": tensor}}
        self._original_params = {} 
        self._applied_loras = set()
        logger.info("LoRAManager initialized.")

    def apply_lora(self, lora_name, weight_slider_val, target_modules):
        if not lora_name or not target_modules:
            return
        lora_path = os.path.join(LORA_DIR, lora_name)
        if not os.path.exists(lora_path):
            logger.error(f"LoRA file not found: {lora_path}")
            return

        logger.info(f"Applying LoRA '{lora_name}' with weight {weight_slider_val} to {target_modules}")
        try:
            raw_lora_tensors = load_file(lora_path, device="cpu")
        except Exception as e:
            logger.error(f"Failed to load LoRA file {lora_path}: {e}", exc_info=True)
            return

        for target_key in target_modules:
            map_info = LORA_TARGET_MAP.get(target_key)
            if not map_info: continue

            # Ensure transformer is loaded if it's a target
            if map_info["model_key"] == "transformer":
                shared_state.models['transformer'] = model_loader.get_transformer_model()

            model = shared_state.models.get(map_info["model_key"])
            if not model: continue
            
            lora_tensors = _translate_lora_state_dict(raw_lora_tensors, map_info["lora_prefix"])
            self._merge_model_statically(model, lora_tensors, map_info["lora_prefix"], float(weight_slider_val), target_key)

        self._applied_loras.add(lora_name)

    def _merge_model_statically(self, model, lora_tensors, lora_prefix, multiplier, model_target_key):
        modified_keys_count = 0

        # --- Build a reliable map from LoRA key names to the model's actual parameter keys ---
        model_sd = model.state_dict()
        lora_to_model_key_map = {}
        for model_key in model_sd:
            if model_key.endswith(".weight"):
                # Convert model key 'path.to.layer.weight' to LoRA key 'prefix_path_to_layer'
                lora_module_name = f"{lora_prefix}_{model_key.rsplit('.', 1)[0].replace('.', '_')}"
                lora_to_model_key_map[lora_module_name] = model_key

        # --- Iterate through LoRA weights and apply them ---
        lora_down_keys = [k for k in lora_tensors.keys() if 'lora_down.weight' in k and k.startswith(lora_prefix)]

        for down_key in lora_down_keys:
            up_key = down_key.replace('lora_down.weight', 'lora_up.weight')
            alpha_key = down_key.replace('lora_down.weight', 'alpha')

            if up_key not in lora_tensors: continue

            # Find the corresponding parameter key in the model's state_dict
            lora_module_name = down_key.rsplit('.lora_down.weight', 1)[0]
            model_weight_key = lora_to_model_key_map.get(lora_module_name)

            if not model_weight_key:
                logger.debug(f"Skipping LoRA key (no match in model): {down_key}")
                continue

            # Get the actual layer module from the model
            module_path = model_weight_key.rsplit('.weight', 1)[0]
            try:
                target_layer = _get_module(model, module_path)
            except AttributeError:
                logger.warning(f"Could not find module '{module_path}' in model for LoRA key '{down_key}'.")
                continue

            # --- Perform the merge ---
            try:
                device, dtype = target_layer.weight.device, target_layer.weight.dtype

                # Backup the original weight if not already backed up
                if model_target_key not in self._original_params: self._original_params[model_target_key] = {}
                if module_path not in self._original_params[model_target_key]:
                    self._original_params[model_target_key][module_path] = target_layer.weight.clone()

                # Calculate scale
                rank = lora_tensors[down_key].shape[0]
                alpha = lora_tensors.get(alpha_key, torch.tensor(float(rank))).item()
                scale = alpha / rank if rank > 0 else 1.0

                down_w = lora_tensors[down_key].to(device, dtype=dtype)
                up_w = lora_tensors[up_key].to(device, dtype=dtype)
                delta_w = None

                if isinstance(target_layer, nn.Linear):
                    delta_w = (up_w @ down_w)
                elif isinstance(target_layer, nn.Conv2d):
                    if down_w.size()[2:4] == (1, 1): # Conv2d 1x1
                        delta_w = (up_w.squeeze(3).squeeze(2) @ down_w.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                    else: # Conv2d 3x3 or other kernel sizes
                        delta_w = F.conv2d(down_w.permute(1, 0, 2, 3), up_w).permute(1, 0, 2, 3)
                
                if delta_w is not None:
                    merged_weight = (target_layer.weight.data.to(torch.float32) + (multiplier * scale * delta_w).to(torch.float32)).to(dtype)
                    target_layer.weight = nn.Parameter(merged_weight, requires_grad=False)
                    modified_keys_count += 1
                else:
                    logger.warning(f"Unsupported layer type for LoRA merge: {type(target_layer)} for key {module_path}")

            except Exception as e:
                logger.error(f"Failed to merge weights for module {module_path}: {e}", exc_info=True)
        
        if modified_keys_count > 0:
            logger.info(f"Successfully merged weights into {modified_keys_count} layers.")
        else:
            logger.warning("No layers were merged. Check LoRA compatibility and DEBUG logs.")

    def revert_all_loras(self):
        if not self._original_params: return
        logger.info(f"Reverting layers for LoRAs: {self._applied_loras}")
        
        reverted_count = 0
        for model_key, params_to_revert in self._original_params.items():
            model = shared_state.models.get(model_key)
            if not model:
                logger.warning(f"Model '{model_key}' not found for reversion, skipping.")
                continue

            for module_path, original_param in params_to_revert.items():
                try:
                    # The module_path is the full path to the layer itself (e.g., 'transformer_blocks.0.attn.to_q').
                    # We need to get the layer module, not its parent.
                    target_layer = _get_module(model, module_path)
                    # The original_param is a raw tensor, so it must be wrapped in nn.Parameter
                    # to correctly restore the weight of the layer.
                    target_layer.weight = nn.Parameter(original_param, requires_grad=False)
                    reverted_count += 1
                except Exception as e:
                    logger.error(f"Failed to revert parameter for module {module_path} in model {model_key}: {e}", exc_info=True)
        
        logger.info(f"Reverted {reverted_count} total layers across {len(self._original_params)} models.")
        self._original_params.clear()
        self._applied_loras.clear()

def handle_lora_upload_and_update_ui(app_state, uploaded_file):
    if uploaded_file is None:
        return app_state, "", gr.update(visible=False), "", 1.0, []
    
    lora_name = os.path.basename(uploaded_file.name)
    persistent_path = os.path.join(LORA_DIR, lora_name)
    
    os.makedirs(os.path.dirname(persistent_path), exist_ok=True)
    shutil.copy(uploaded_file.name, persistent_path)
    
    logger.info(f"Saved LoRA file to: {persistent_path}")
    
    # Clear any previously loaded LoRA state to ensure a fresh start
    if "lora_state" in app_state and "loaded_loras" in app_state["lora_state"]:
        app_state["lora_state"]["loaded_loras"].clear()
        
    app_state.setdefault("lora_state", {}).setdefault("loaded_loras", {})[lora_name] = {"path": persistent_path}
    
    gr.Info(f"Loaded '{lora_name}'.")
    return app_state, lora_name, gr.update(visible=True), gr.update(value=lora_name), gr.update(value=0.8), gr.update(value=["transformer"])