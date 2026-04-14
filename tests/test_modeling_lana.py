from types import SimpleNamespace

import torch

from lana_radgen import LanaForConditionalGeneration
from lana_radgen.attention.layerwise_anatomical_attention import build_layerwise_attention_bias, build_legacy_gaussian_attention_bias


def test_layerwise_attention_bias_shape():
    masks = torch.ones(2, 1, 512, 512)
    bias = build_layerwise_attention_bias(masks, num_layers=4, target_tokens=64)
    assert bias.shape == (2, 4, 64, 64)


def test_legacy_gaussian_attention_bias_shape():
    masks = torch.ones(2, 1, 512, 512)
    bias = build_legacy_gaussian_attention_bias(masks, num_layers=4, target_query_tokens=128, target_key_tokens=96)
    assert bias.shape == (2, 4, 128, 96)


def test_gaussian_legacy_model_bias_expands_to_full_sequence_width():
    model = LanaForConditionalGeneration.__new__(LanaForConditionalGeneration)
    model.config = SimpleNamespace(
        attention_bias_mode="gaussian_legacy",
        num_attention_layers=4,
        layer_mask_base_kernel_size=3,
        layer_mask_kernel_growth=2,
        anatomical_attention_bias=2.0,
    )
    masks = torch.ones(2, 1, 512, 512)
    bias = model._build_layerwise_bias(masks, total_sequence_length=130, vision_prefix_length=96)
    assert bias.shape == (2, 4, 130, 130)
    assert torch.count_nonzero(bias[:, :, :, 96:]) == 0
