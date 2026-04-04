from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers.utils import ModelOutput


@dataclass
class LanaModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    layerwise_attentions: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    vision_features: Optional[torch.FloatTensor] = None
