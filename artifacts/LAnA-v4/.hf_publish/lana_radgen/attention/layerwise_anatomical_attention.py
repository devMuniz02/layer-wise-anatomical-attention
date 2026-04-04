import torch
import torch.nn.functional as F


def _gaussian_kernel_1d(kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    radius = kernel_size // 2
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


@torch.no_grad()
def build_layerwise_attention_bias(
    masks: torch.Tensor,
    num_layers: int,
    target_tokens: int,
    base_kernel_size: int = 3,
    kernel_growth: int = 2,
    strength: float = 2.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    if masks.ndim == 3:
        masks = masks.unsqueeze(1)
    if masks.ndim != 4 or masks.shape[1] != 1:
        raise ValueError(f"Expected masks shaped (B,1,H,W) or (B,H,W), got {tuple(masks.shape)}")

    masks = masks.float()
    batch_size = masks.shape[0]
    resized = F.interpolate(masks, size=(32, 32), mode="bilinear", align_corners=False).clamp(0.0, 1.0)

    max_kernel = base_kernel_size + max(num_layers, 0) * kernel_growth
    if max_kernel % 2 == 0:
        max_kernel += 1
    pad = max_kernel // 2

    weight_h = torch.zeros((num_layers, 1, 1, max_kernel), device=resized.device, dtype=resized.dtype)
    weight_v = torch.zeros((num_layers, 1, max_kernel, 1), device=resized.device, dtype=resized.dtype)

    for layer_idx in range(num_layers):
        kernel_size = base_kernel_size + (num_layers - layer_idx) * kernel_growth
        if kernel_size % 2 == 0:
            kernel_size += 1
        sigma = max((kernel_size - 1) / 6.0, 1e-3)
        kernel = _gaussian_kernel_1d(kernel_size, sigma, resized.device, resized.dtype)
        start = (max_kernel - kernel_size) // 2
        end = start + kernel_size
        weight_h[layer_idx, 0, 0, start:end] = kernel
        weight_v[layer_idx, 0, start:end, 0] = kernel

    repeated = resized.expand(batch_size, num_layers, 32, 32).contiguous()
    horizontal = F.conv2d(F.pad(repeated, (pad, pad, 0, 0), mode="reflect"), weight_h, groups=num_layers)
    vertical = F.conv2d(F.pad(horizontal, (0, 0, pad, pad), mode="reflect"), weight_v, groups=num_layers)

    min_vals = vertical.amin(dim=(2, 3), keepdim=True)
    max_vals = vertical.amax(dim=(2, 3), keepdim=True)
    normalized = (vertical - min_vals) / (max_vals - min_vals).clamp_min(eps)

    flat = normalized.view(batch_size, num_layers, -1)
    if flat.shape[-1] != target_tokens:
        flat = F.interpolate(flat, size=target_tokens, mode="linear", align_corners=False)
    layerwise_bias = flat.unsqueeze(-2).expand(-1, -1, target_tokens, -1)
    return torch.tril(layerwise_bias) * strength
