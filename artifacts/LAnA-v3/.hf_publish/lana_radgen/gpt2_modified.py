from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Model
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block, eager_attention_forward


class GPT2AttentionModified(GPT2Attention):
    def forward(
        self,
        hidden_states: Optional[tuple[torch.FloatTensor]],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ):
        is_cross_attention = encoder_hidden_states is not None
        if past_key_values is not None:
            if isinstance(past_key_values, EncoderDecoderCache):
                is_updated = past_key_values.is_updated.get(self.layer_idx)
                curr_past_key_value = past_key_values.cross_attention_cache if is_cross_attention else past_key_values.self_attention_cache
            else:
                curr_past_key_value = past_key_values

        if is_cross_attention:
            if not hasattr(self, "q_attn"):
                raise ValueError("Cross-attention requires q_attn to be defined.")
            query_states = self.q_attn(hidden_states)
            attention_mask = encoder_attention_mask
            if past_key_values is not None and is_updated:
                key_states = curr_past_key_value.layers[self.layer_idx].keys
                value_states = curr_past_key_value.layers[self.layer_idx].values
            else:
                key_states, value_states = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
                shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
                key_states = key_states.view(shape_kv).transpose(1, 2)
                value_states = value_states.view(shape_kv).transpose(1, 2)
        else:
            query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
            shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
            key_states = key_states.view(shape_kv).transpose(1, 2)
            value_states = value_states.view(shape_kv).transpose(1, 2)

        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        query_states = query_states.view(shape_q).transpose(1, 2)

        if (past_key_values is not None and not is_cross_attention) or (
            past_key_values is not None and is_cross_attention and not is_updated
        ):
            cache_position = cache_position if not is_cross_attention else None
            key_states, value_states = curr_past_key_value.update(
                key_states, value_states, self.layer_idx, {"cache_position": cache_position}
            )
            if is_cross_attention:
                past_key_values.is_updated[self.layer_idx] = True

        is_causal = attention_mask is None and query_states.shape[-2] > 1 and not is_cross_attention
        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            head_mask=head_mask,
            dropout=self.attn_dropout.p if self.training else 0.0,
            is_causal=is_causal,
            **kwargs,
        )

        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        return attn_output, attn_weights


class GPT2BlockModified(GPT2Block):
    def __init__(self, config, layer_idx=None):
        super().__init__(config=config, layer_idx=layer_idx)
        self.attn = GPT2AttentionModified(config=config, layer_idx=layer_idx)


class GPT2ModelModified(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        self.config_causal = config
        self.config_causal._attn_implementation = "eager"
        self.h = nn.ModuleList([GPT2BlockModified(config, layer_idx=i) for i in range(config.num_hidden_layers)])

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[tuple[tuple[torch.Tensor]], Cache]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        segmentation_mask: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Union[tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache()
            elif isinstance(past_key_values, tuple):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            if self.config.add_cross_attention and not isinstance(past_key_values, EncoderDecoderCache):
                past_key_values = EncoderDecoderCache(past_key_values, DynamicCache())

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)

        if attention_mask is not None and attention_mask.ndim < 4:
            attention_mask = attention_mask.view(batch_size, -1)

        causal_mask = create_causal_mask(
            config=self.config_causal,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        _use_sdpa = self._attn_implementation == "sdpa" and output_attentions is False and head_mask is None
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            if _use_sdpa:
                encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    mask=encoder_attention_mask, dtype=inputs_embeds.dtype, tgt_len=input_shape[-1]
                )
            elif self._attn_implementation != "flash_attention_2":
                encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        if head_mask is None:
            head_mask = [None] * self.config.n_layer

        if token_type_ids is not None:
            hidden_states = hidden_states + self.wte(token_type_ids)

        hidden_states = self.drop(hidden_states)
        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None

        for i, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            block_mask = causal_mask
            if segmentation_mask is not None and causal_mask is not None:
                block_mask = causal_mask.clone()
                seq_len = input_shape[-1]
                if block_mask.shape[2] != seq_len or block_mask.shape[3] != seq_len:
                    block_mask = block_mask[:, :, :seq_len, :seq_len]
                layer_bias = segmentation_mask[:, i, : block_mask.shape[2], : block_mask.shape[3]].unsqueeze(1)
                block_mask = block_mask + layer_bias.to(dtype=block_mask.dtype, device=block_mask.device)

            outputs = block(
                hidden_states=hidden_states,
                past_key_values=past_key_values if not (self.gradient_checkpointing and self.training) else None,
                cache_position=cache_position,
                attention_mask=block_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                head_mask=head_mask[i],
                **kwargs,
            )
            if isinstance(outputs, tuple):
                hidden_states = outputs[0]
                if output_attentions and len(outputs) > 1:
                    all_self_attentions = all_self_attentions + (outputs[1],)
                    if self.config.add_cross_attention and len(outputs) > 2:
                        all_cross_attentions = all_cross_attentions + (outputs[2],)
            else:
                hidden_states = outputs

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        past_key_values = past_key_values if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attentions, all_cross_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class GPT2LMHeadModelModified(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2ModelModified(config)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple[tuple[torch.Tensor]]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        segmentation_mask: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Union[tuple, CausalLMOutputWithCrossAttentions]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            cache_position=cache_position,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            segmentation_mask=segmentation_mask,
            **kwargs,
        )
        hidden_states = transformer_outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep > 0 else slice(None)
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )


@torch.no_grad()
def expand_gpt2_positional_embeddings(
    model: torch.nn.Module,
    new_max_positions: int,
    mode: str = "linear",
    align_corners: bool = True,
):
    if hasattr(model, "transformer") and hasattr(model.transformer, "wpe"):
        model_for_wpe = model.transformer
    elif hasattr(model, "wpe"):
        model_for_wpe = model
    else:
        raise ValueError("Model does not expose GPT-2 positional embeddings.")

    wpe = model_for_wpe.wpe
    old_n, d = wpe.weight.shape
    if new_max_positions == old_n:
        return model

    device = wpe.weight.device
    dtype = wpe.weight.dtype
    if new_max_positions < old_n:
        new_weight = wpe.weight[:new_max_positions].clone()
    else:
        if mode != "linear":
            raise ValueError(f"Unsupported positional expansion mode: {mode}")
        w = wpe.weight.transpose(0, 1).unsqueeze(0)
        w_new = F.interpolate(w, size=new_max_positions, mode="linear", align_corners=align_corners)
        new_weight = w_new.squeeze(0).transpose(0, 1).contiguous()

    new_wpe = torch.nn.Embedding(new_max_positions, d, device=device, dtype=dtype)
    new_wpe.weight.copy_(new_weight)
    if hasattr(model, "transformer") and hasattr(model.transformer, "wpe"):
        model.transformer.wpe = new_wpe
    else:
        model.wpe = new_wpe
    if hasattr(model.config, "n_positions"):
        model.config.n_positions = new_max_positions
    if hasattr(model.config, "n_ctx"):
        model.config.n_ctx = new_max_positions
    return model


def create_decoder(text_model_name: str, attention_implementation: str, max_position_embeddings: int, **decoder_kwargs):
    config = GPT2Config.from_pretrained(text_model_name)
    config._attn_implementation = attention_implementation
    config.n_positions = max_position_embeddings
    config.n_ctx = max_position_embeddings
    config.use_cache = decoder_kwargs.pop("use_cache", True)
    decoder = GPT2LMHeadModelModified.from_pretrained(text_model_name, config=config, **decoder_kwargs)
    decoder.config._attn_implementation = attention_implementation
    return expand_gpt2_positional_embeddings(decoder, new_max_positions=max_position_embeddings, mode="linear")
