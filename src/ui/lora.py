# ui/lora.py (REVISED)
# Manages LoRA uploads and the dynamic application/reversion of LoRA layers.
# Corrected the mathematical formula in the LoRA forward pass.
# REVISION: Replaced dynamic layer patching with static weight merging to resolve
# severe performance degradation caused by breaking fused kernels. The dynamic
# patching code is retained for future reference.
# REVISION 2: Implemented a numerically stable merge process (FP16 matmul, FP32 add)
# to prevent denormal floats that were causing a persistent per-iteration slowdown.

import os
import torch
import math
import torch.nn as nn
import gradio as gr
import shutil
import logging
from safetensors.torch import load_file

from .enums import ComponentKey as K
from . import shared_state

# --- Setup Logger ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

LORA_DIR = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'loras')))
os.makedirs(LORA_DIR, exist_ok=True)

LORA_TARGET_MAP = {
    "transformer": {"model_key": "transformer", "lora_prefix": "lora_unet"},
}

# --- NOTE: The LoRALinearLayer is no longer used in the default static merging workflow. ---
# It is preserved here for reference or for potential future use in an optimized
# high-performance dynamic LoRA implementation.
class LoRALinearLayer(nn.Module):
    def __init__(self, original_layer: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.original_layer = original_layer
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.scale = alpha / rank if rank > 0 else 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_device = self.original_layer.weight.device
        original_dtype = self.original_layer.weight.dtype
        
        x_dev = x.to(device=self.lora_A.device, dtype=self.lora_A.dtype)

        original_output = self.original_layer(x.to(device=original_device, dtype=original_dtype))
        
        if self.scale > 0:
            lora_delta = (x_dev @ self.lora_A.T) @ self.lora_B.T
            scaled_delta = (lora_delta * self.scale).to(device=original_device, dtype=original_output.dtype)
            return original_output + scaled_delta
            
        return original_output

def _find_and_set_module(model, module_key, new_module):
    tokens = module_key.split('.')
    parent_key = '.'.join(tokens[:-1])
    child_key = tokens[-1]
    
    # Traverse the model hierarchy to find the parent module.
    parent_module = model
    for part in parent_key.split('.'):
        if part.isdigit():
            parent_module = parent_module[int(part)]
        else:
            parent_module = getattr(parent_module, part)
            
    # Set the new module on the parent.
    setattr(parent_module, child_key, new_module)

def _get_module(model, module_key):
    """Helper function to retrieve a module from a model using its key."""
    parent_module = model
    for part in module_key.split('.'):
        if part.isdigit():
            parent_module = parent_module[int(part)]
        else:
            parent_module = getattr(parent_module, part)
    return parent_module


def _convert_hunyuan_keys_to_framepack(lora_sd: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    logger.info("Hunyuan LoRA detected, attempting to convert keys to FramePack format...")
    new_lora_sd = {}
    
    name_map = {
        "double_blocks": "transformer_blocks",
        "img_attn_qkv": "attn_to_QKV",
        "img_attn_proj": "attn_to_out_0",
        "img_mlp.fc1": "ff_net.0.proj",
        "img_mlp.fc2": "ff_net.2",
        "txt_attn_qkv": "attn_add_QKV_proj",
        "txt_attn_proj": "attn_to_add_out",
        "txt_mlp.fc1": "ff_context_net.0.proj",
        "txt_mlp.fc2": "ff_context_net.2",
    }

    for key, weight in lora_sd.items():
        original_key = key
        
        if key.startswith("transformer."):
            key = key[len("transformer."):]
        
        translated_key = key
        for source, target in name_map.items():
            translated_key = translated_key.replace(source, target)

        if translated_key == key and "double_blocks" in key:
            logger.debug(f"Skipping untranslated key: '{original_key}'")
            continue

        if ".lora_A.weight" in translated_key:
            module_path = translated_key.replace(".lora_A.weight", "")
            new_key = f"{prefix}_{module_path.replace('.', '_')}.lora_down.weight"
        elif ".lora_B.weight" in translated_key:
            module_path = translated_key.replace(".lora_B.weight", "")
            new_key = f"{prefix}_{module_path.replace('.', '_')}.lora_up.weight"
        else:
            new_lora_sd[translated_key] = weight
            continue
        
        logger.debug(f"Key translation: '{original_key}' -> '{new_key}'")

        if "QKV" in new_key:
            qkv_dim = 3072 
            key_q, key_k, key_v = new_key.replace("QKV", "q"), new_key.replace("QKV", "k"), new_key.replace("QKV", "v")
            if "lora_down" in new_key:
                new_lora_sd[key_q], new_lora_sd[key_k], new_lora_sd[key_v] = weight, weight, weight
            elif "lora_up" in new_key:
                if weight.size(0) == qkv_dim * 3:
                    new_lora_sd[key_q] = weight[:qkv_dim]
                    new_lora_sd[key_k] = weight[qkv_dim:qkv_dim * 2]
                    new_lora_sd[key_v] = weight[qkv_dim * 2:]
                else:
                    logger.warning(f"Skipping QKV split for key '{original_key}' due to unexpected dimension: {weight.size()}. Expected dimension 0 to be {qkv_dim * 3}.")
                    continue
            else:
                 new_lora_sd[new_key] = weight
        else:
            new_lora_sd[new_key] = weight
            
    return new_lora_sd

def _translate_lora_state_dict(lora_sd: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    keys = list(lora_sd.keys())
    if not keys: return lora_sd

    is_hunyuan_format = any("double_blocks" in k or "single_blocks" in k for k in keys)
    if is_hunyuan_format:
        return _convert_hunyuan_keys_to_framepack(lora_sd, prefix)
    
    logger.info("LoRA does not appear to be in a known incompatible Hunyuan format. Applying keys as-is.")
    return lora_sd

class LoRAManager:
    def __init__(self):
        self._original_weights = {}
        self._applied_loras = set()
        logger.info("LoRAManager initialized.")

    def apply_lora(self, lora_name, weight_slider_val, target_modules, merge_statically=True):
        if not lora_name or not target_modules:
            logger.warning("No LoRA name or target modules provided. Skipping.")
            return

        lora_path = os.path.join(LORA_DIR, lora_name)
        if not os.path.exists(lora_path):
            logger.error(f"LoRA file not found at {lora_path}")
            return

        logger.info(f"Applying LoRA '{lora_name}' with weight {weight_slider_val} to {target_modules}")

        try:
            raw_lora_tensors = load_file(lora_path, device="cpu")
        except Exception as e:
            logger.error(f"Failed to load LoRA file {lora_path}: {e}", exc_info=True)
            return
        
        for target_key in target_modules:
            map_info = LORA_TARGET_MAP.get(target_key)
            if not map_info:
                logger.warning(f"Unknown LoRA target '{target_key}'. Skipping.")
                continue

            model = shared_state.models.get(map_info["model_key"])
            lora_prefix = map_info["lora_prefix"]
            if not model:
                logger.warning(f"Model for '{target_key}' not found. Skipping.")
                continue

            lora_tensors = _translate_lora_state_dict(raw_lora_tensors, lora_prefix)
            
            rank, alpha = 4, float(weight_slider_val)
            for key, tensor in lora_tensors.items():
                if "lora_down.weight" in key and len(tensor.shape) == 2:
                    alpha_key = key.replace("lora_down.weight", "alpha")
                    if alpha_key in lora_tensors:
                        alpha = lora_tensors[alpha_key].item() * float(weight_slider_val)
                    rank = tensor.shape[0]
                    break 
            logger.info(f"Determined LoRA rank: {rank}, Alpha: {alpha}")

            if merge_statically:
                self._merge_model_statically(model, lora_tensors, lora_prefix, rank, alpha)
            else:
                self._patch_model(model, lora_tensors, lora_prefix, rank, alpha)

        self._applied_loras.add(lora_name)

    def _merge_model_statically(self, model, lora_tensors, lora_prefix, rank, alpha):
        """
        Merges LoRA weights directly into the model's linear layers.
        This version uses a numerically stable calculation path to avoid creating
        denormal floats that cause inference slowdowns.
        """
        modified_keys_count = 0
        scale = alpha / rank if rank > 0 else 0.0

        lora_layers_map = {}
        for key, tensor in lora_tensors.items():
            if not key.startswith(lora_prefix) or (".lora_down" not in key and ".lora_up" not in key):
                continue
            
            base_name = key.replace(f'{lora_prefix}_', '').split('.lora_')[0]
            
            if base_name not in lora_layers_map: lora_layers_map[base_name] = {}
            if "lora_down.weight" in key: lora_layers_map[base_name]["down"] = tensor
            elif "lora_up.weight" in key: lora_layers_map[base_name]["up"] = tensor

        model_linear_layers = {name: mod for name, mod in model.named_modules() if isinstance(mod, nn.Linear)}
        logger.debug(f"Found {len(model_linear_layers)} linear layers in model {model.__class__.__name__}.")

        for lora_base_name, tensors in lora_layers_map.items():
            if "up" not in tensors or "down" not in tensors: continue

            found_model_key = None
            for model_key_name in model_linear_layers.keys():
                if model_key_name.replace('.', '_') == lora_base_name:
                    found_model_key = model_key_name
                    break

            if found_model_key:
                try:
                    target_layer = model_linear_layers[found_model_key]
                    original_dtype = target_layer.weight.dtype
                    device = target_layer.weight.device

                    # --- Backup the original weight if not already done ---
                    if found_model_key not in self._original_weights:
                        self._original_weights[found_model_key] = target_layer.weight.data.clone()

                    # --- Numerically Stable Merge Calculation ---
                    # 1. Cast LoRA weights to float16 for performant matrix multiplication.
                    lora_down = tensors["down"].to(device, dtype=torch.float16)
                    lora_up = tensors["up"].to(device, dtype=torch.float16)

                    # 2. Calculate the delta matrix in float16 and apply scaling.
                    delta_w = (lora_up @ lora_down) * scale

                    # 3. Add the delta to the original weight in float32 to preserve precision,
                    #    then cast back to the original dtype before applying.
                    merged_weight = (target_layer.weight.data.to(torch.float32) + delta_w.to(torch.float32)).to(original_dtype)
                    
                    # --- Apply the new weight to the layer ---
                    target_layer.weight.data.copy_(merged_weight)
                    
                    logger.debug(f"Successfully merged weights for '{found_model_key}' from LoRA key '{lora_base_name}'")
                    modified_keys_count += 1
                except Exception as e:
                    logger.error(f"Failed to merge weights for module {found_model_key}: {e}", exc_info=True)
            else:
                logger.debug(f"No matching layer in model for LoRA key: '{lora_base_name}'")

        if modified_keys_count > 0:
            logger.info(f"Successfully merged weights into {modified_keys_count} layers in {model.__class__.__name__}.")
        else:
            logger.warning("No layers were merged. Check LoRA compatibility and DEBUG logs for key mismatches.")


    def _patch_model(self, model, lora_tensors, lora_prefix, rank, alpha):
        """
        DEPRECATED: This method dynamically replaces layers with LoRALinearLayer.
        It is preserved for reference but is not recommended due to performance issues.
        """
        logger.info("Using dynamic LoRA patching (performance may be degraded).")
        modified_keys_count = 0
        
        lora_layers_map = {}
        for key, tensor in lora_tensors.items():
            if not key.startswith(lora_prefix) or (".lora_down" not in key and ".lora_up" not in key):
                continue
            
            base_name = key.replace(f'{lora_prefix}_', '').split('.lora_')[0]
            
            if base_name not in lora_layers_map: lora_layers_map[base_name] = {}
            if "lora_down.weight" in key: lora_layers_map[base_name]["down"] = tensor
            elif "lora_up.weight" in key: lora_layers_map[base_name]["up"] = tensor

        model_linear_layers = {name: mod for name, mod in model.named_modules() if isinstance(mod, nn.Linear)}
        logger.debug(f"Found {len(model_linear_layers)} linear layers in model {model.__class__.__name__}.")

        for lora_base_name, tensors in lora_layers_map.items():
            if "up" not in tensors or "down" not in tensors: continue

            found_model_key = None
            for model_key_name in model_linear_layers.keys():
                if model_key_name.replace('.', '_') == lora_base_name:
                    found_model_key = model_key_name
                    break

            if found_model_key:
                try:
                    original_layer = model_linear_layers[found_model_key]
                    new_layer = LoRALinearLayer(original_layer, rank, alpha)
                    new_layer.lora_A.data.copy_(tensors["down"])
                    new_layer.lora_B.data.copy_(tensors["up"])
                    _find_and_set_module(model, found_model_key, new_layer)
                    logger.debug(f"Successfully patched '{found_model_key}' from LoRA key '{lora_base_name}'")
                    modified_keys_count += 1
                except Exception as e:
                    logger.error(f"Failed to patch module {found_model_key}: {e}", exc_info=True)
            else:
                logger.debug(f"No matching layer in model for LoRA key: '{lora_base_name}'")

        if modified_keys_count > 0:
            logger.info(f"Successfully replaced {modified_keys_count} layers in {model.__class__.__name__}.")
        else:
            logger.warning("No layers were replaced. Check LoRA compatibility and DEBUG logs for key mismatches.")

    def revert_all_loras(self):
        if not self._original_weights: 
            logger.info("No LoRAs to revert.")
            return

        logger.info(f"Reverting statically merged weights for LoRAs: {self._applied_loras}")
        model = shared_state.models.get("transformer")
        if model is None:
            logger.warning("Transformer model not found for reversion.")
            return

        reverted_count = 0
        for module_key, original_weight in self._original_weights.items():
            try:
                module_to_revert = _get_module(model, module_key)
                module_to_revert.weight.data.copy_(original_weight)
                reverted_count += 1
            except Exception as e:
                logger.error(f"Failed to revert weights for module {module_key}: {e}", exc_info=True)
        
        logger.info(f"Reverted weights for {reverted_count} total layers.")
        self._original_weights.clear()
        self._applied_loras.clear()

def handle_lora_upload_and_update_ui(app_state, uploaded_file):
    if uploaded_file is None:
        return app_state, "", gr.update(visible=False), "", 1.0, []
    lora_name = os.path.basename(uploaded_file.name)
    persistent_path = os.path.join(LORA_DIR, lora_name)
    shutil.move(uploaded_file.name, persistent_path)
    logger.info(f"Saved LoRA file to: {persistent_path}")
    app_state.get("lora_state", {}).get("loaded_loras", {}).clear()
    app_state.setdefault("lora_state", {}).setdefault("loaded_loras", {})[lora_name] = {"path": persistent_path}
    gr.Info(f"Loaded '{lora_name}'.")
    return app_state, lora_name, gr.update(visible=True), gr.update(value=lora_name), gr.update(value=0.8), gr.update(value=["transformer"])