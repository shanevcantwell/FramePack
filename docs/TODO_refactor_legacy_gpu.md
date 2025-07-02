# TODO: Refactor for Legacy GPU (<SM80, Turing and Older) Support

## Problem Statement

On Turing and earlier GPUs, the Hunyuan Packed model's default BF16 precision is not supported by CUDA. This causes PyTorch to fall back to the "slow_conv3d_forward" kernel, which is not implemented for CUDA, resulting in runtime errors such as:

```
NotImplementedError: Could not run 'aten::slow_conv3d_forward' with arguments from the 'CUDA' backend. ...
```

## Root Cause

- Turing and earlier GPUs do **not** support BF16 on CUDA.
- PyTorch attempts to run Conv3d in BF16, fails, and falls back to a CPU-only kernel that is not available on CUDA.
- This can also occur if tensors are not contiguous or not on the correct device.

## Solution Overview

**Force all tensors and Conv3d weights to FP32 before any Conv3d operation in the model.**
- This avoids the unsupported kernel and ensures correct execution on legacy GPUs.
- This should be done in the `forward` methods of `HunyuanVideoPatchEmbed` and `HunyuanVideoPatchEmbedForCleanLatents`.

## Implementation Steps

### 1. Patch Conv3d Input Handling

In `diffusers_helper/models/hunyuan_video_packed.py`:

#### a. HunyuanVideoPatchEmbed

```python
class HunyuanVideoPatchEmbed(nn.Module):
    def __init__(self, patch_size, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # Force input to float32, contiguous, and on the same device as weights
        x = x.to(dtype=torch.float32, device=self.proj.weight.device)
        if not x.is_contiguous():
            x = x.contiguous()
        return self.proj(x)
```

#### b. HunyuanVideoPatchEmbedForCleanLatents

```python
class HunyuanVideoPatchEmbedForCleanLatents(nn.Module):
    def __init__(self, inner_dim):
        super().__init__()
        self.proj = nn.Conv3d(16, inner_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.proj_2x = nn.Conv3d(16, inner_dim, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        self.proj_4x = nn.Conv3d(16, inner_dim, kernel_size=(4, 8, 8), stride=(4, 8, 8))

    def forward(self, x, proj_type='proj'):
        proj_layer = getattr(self, proj_type)
        x = x.to(dtype=torch.float32, device=proj_layer.weight.device)
        if not x.is_contiguous():
            x = x.contiguous()
        return proj_layer(x)
```

### 2. (Optional) Add Device/Precision Utility

If you have other custom Conv3d modules, consider a utility function to ensure dtype and device alignment.

### 3. No UI/Config Changes Needed

- The backend already forces FP32 and hides the UI toggle for legacy GPUs.
- No changes needed in `layout.py` or shared state logic.

## Testing

- Test on a Turing or earlier GPU.
- Confirm that video generation runs without "slow_conv3d_forward" errors.
- Confirm that output is numerically stable and matches FP32 expectations.

## References

- See error trace in `core.generation_core` and `hunyuan_video_packed.py`.
- See PyTorch docs on [Conv3d](https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html) and CUDA/BF16 support.

---

**Summary:**  
Force all Conv3d inputs and weights to FP32 in the relevant modules to ensure legacy GPU compatibility. No UI or config changes are