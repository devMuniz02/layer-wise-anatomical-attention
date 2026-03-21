import torch

from lana_radgen.attention.layerwise_anatomical_attention import build_layerwise_attention_bias


def test_layerwise_attention_bias_shape():
    masks = torch.ones(2, 1, 512, 512)
    bias = build_layerwise_attention_bias(masks, num_layers=4, target_tokens=64)
    assert bias.shape == (2, 4, 64, 64)
