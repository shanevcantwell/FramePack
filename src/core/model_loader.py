# core/model_loader.py
import torch
import gc
from ui import shared_state
from diffusers import AutoencoderKLHunyuanVideo
from transformers import (
    LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer,
    SiglipImageProcessor, SiglipVisionModel
)
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked # type: ignore
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

def _load_transformer_model():
    """Helper to load and configure the transformer model."""
    print("Loading transformer ...")
    transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(
        "lllyasviel/FramePackI2V_HY", torch_dtype=torch.bfloat16
    ).cpu()
    transformer.eval()
    # In legacy mode, this setting is not optional; it's required for stability.
    # Otherwise, it's controlled by the UI.
    if shared_state.system_info.get('is_legacy_gpu', False):
        print("Legacy GPU: Forcing high quality FP32 transformer output for stability.")
    transformer.high_quality_fp32_output_for_inference = True
    transformer.to(dtype=torch.bfloat16)
    transformer.requires_grad_(False)
    print("Transformer loaded.")
    return transformer

def load_and_configure_models():
    """
    Detects GPU capability, loads the appropriate models, configures them for the
    detected hardware (VRAM, dtype), and populates the shared_state.
    """
    print("Initializing models...")

    # The main HunyuanVideoTransformer3DModelPacked has fallbacks for older GPUs.
    try:
        major_capability, _ = torch.cuda.get_device_capability()
        if major_capability < 8:
            print(f"Legacy GPU detected (Compute Capability {major_capability}.x). Activating compatibility mode.")
            shared_state.system_info['is_legacy_gpu'] = True
    except Exception as e:
        print(f"Could not determine GPU capability. Error: {e}")

    free_mem_gb = get_cuda_free_memory_gb(gpu)
    high_vram = free_mem_gb > 60
    print(f'Free VRAM {free_mem_gb} GB, High-VRAM Mode: {high_vram}')

    # Populate the shared_state.models dictionary
    shared_state.models.update({
        'text_encoder': LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu(),
        'text_encoder_2': CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu(),
        'tokenizer': LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer', legacy=False),
        'tokenizer_2': CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2', legacy=False),
        'vae': AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu(),
        'feature_extractor': SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor'),
        'image_encoder': SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu(),
        'high_vram': high_vram
    })
    # Initialize transformer as None and load it lazily
    shared_state.models['transformer'] = None
    print("Models loaded to CPU. Configuring...")

    # Configure models based on environment
    for model_name in ['vae', 'text_encoder', 'text_encoder_2', 'image_encoder']:
        shared_state.models[model_name].eval()

    if not high_vram:
        shared_state.models['vae'].enable_slicing()
        shared_state.models['vae'].enable_tiling()

    # Set dtypes, forcing float16 for legacy GPU transformer
    for model_name, dtype in [('vae', torch.float16), ('image_encoder', torch.float16), ('text_encoder', torch.float16), ('text_encoder_2', torch.float16)]:
        shared_state.models[model_name].to(dtype=dtype)
        
    for model_obj in shared_state.models.values():
        if isinstance(model_obj, torch.nn.Module): # Ensure it's a PyTorch module before setting requires_grad
            model_obj.requires_grad_(False)

    # Move initial models to GPU or install DynamicSwap
    # Install memory-saving tools or move models to GPU
    if not high_vram:
        print("Low VRAM mode: Installing DynamicSwap.")
        # DynamicSwapInstaller.install_model(shared_state.models['transformer'], device=gpu) # Transformer is now lazy loaded
        DynamicSwapInstaller.install_model(shared_state.models['text_encoder'], device=gpu)
    else:
        print("High VRAM mode: Moving all models to GPU.")
        for model_name in ['text_encoder', 'text_encoder_2', 'image_encoder', 'vae']: # Removed transformer
            shared_state.models[model_name].to(gpu)

    print("Model configuration and placement complete.")

def get_transformer_model(force_reload=False):
    """Ensures the transformer model is loaded and returns it."""
    if shared_state.models.get('transformer') is None or force_reload:
        if shared_state.models.get('transformer') is not None: # If forcing reload, clear old model
            del shared_state.models['transformer']
            gc.collect()
            torch.cuda.empty_cache()
        shared_state.models['transformer'] = _load_transformer_model()
        if not shared_state.models['high_vram']:
            DynamicSwapInstaller.install_model(shared_state.models['transformer'], device=gpu)
        else:
            shared_state.models['transformer'].to(gpu)
    return shared_state.models['transformer']
