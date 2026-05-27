#    Copyright 2024 Hao Zhang
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

# from ...constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM
from llava.model.language_model.llm_visual_3d_rope import (
    clear_qwen2_visual_3d_rope_context,
    collect_qwen2_visual_3d_rope_stats,
    install_qwen2_visual_3d_rope_attention,
    qwen2_visual_3d_rope_requires_eager,
)

# from .qwen.modeling_qwen import QWenLMHeadModel, QWenModel
# from .qwen.configuration_qwen import QWenConfig


def _as_bool_config(value, default=False):
    if value is None:
        return default
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        install_qwen2_visual_3d_rope_attention()
        super(LlavaQwenModel, self).__init__(config)

    def _decoder_layer_forward_with_llm_geo(
        self,
        decoder_layer,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_values,
        output_attentions,
        use_cache,
        llm_geo_pos,
        llm_geo_mask,
    ):
        attn = getattr(decoder_layer, "self_attn", None)
        if attn is not None:
            attn._llm_visual_3d_rope_pos = llm_geo_pos
            attn._llm_visual_3d_rope_mask = llm_geo_mask
            attn.last_llm_visual_3d_rope_stats = None
        try:
            return decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
        finally:
            if attn is not None:
                for attr in ("_llm_visual_3d_rope_pos", "_llm_visual_3d_rope_mask"):
                    if hasattr(attn, attr):
                        delattr(attn, attr)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        llm_geo_pos: Optional[torch.FloatTensor] = None,
        llm_geo_mask: Optional[torch.BoolTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if qwen2_visual_3d_rope_requires_eager(self.config):
            raise RuntimeError("LLM visual-token 3D RoPE requires Qwen2 eager attention; disable FlashAttention/SDPA.")
        try:
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            use_cache = use_cache if use_cache is not None else self.config.use_cache
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
            elif input_ids is not None:
                batch_size, seq_length = input_ids.shape
            elif inputs_embeds is not None:
                batch_size, seq_length, _ = inputs_embeds.shape
            else:
                raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

            if self.gradient_checkpointing and self.training and use_cache:
                use_cache = False

            past_key_values_length = 0
            if use_cache:
                use_legacy_cache = not isinstance(past_key_values, Cache)
                if use_legacy_cache:
                    past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_key_values_length = past_key_values.get_usable_length(seq_length)
            else:
                use_legacy_cache = False

            if position_ids is None:
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                position_ids = torch.arange(
                    past_key_values_length,
                    seq_length + past_key_values_length,
                    dtype=torch.long,
                    device=device,
                )
                position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
            else:
                position_ids = position_ids.view(-1, seq_length).long()

            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)

            if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
                is_padding_right = attention_mask[:, -1].sum().item() != batch_size
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'. "
                        "Use left padding for batched generation."
                    )

            if self._attn_implementation == "flash_attention_2":
                attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            elif self._attn_implementation == "sdpa" and not output_attentions:
                attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask,
                    (batch_size, seq_length),
                    inputs_embeds,
                    past_key_values_length,
                )
            else:
                attention_mask = _prepare_4d_causal_attention_mask(
                    attention_mask,
                    (batch_size, seq_length),
                    inputs_embeds,
                    past_key_values_length,
                    sliding_window=self.config.sliding_window,
                )

            hidden_states = inputs_embeds
            all_hidden_states = () if output_hidden_states else None
            all_self_attns = () if output_attentions else None
            next_decoder_cache = None

            for decoder_layer in self.layers:
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        self._decoder_layer_forward_with_llm_geo,
                        decoder_layer,
                        hidden_states,
                        attention_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                        llm_geo_pos,
                        llm_geo_mask,
                    )
                else:
                    layer_outputs = self._decoder_layer_forward_with_llm_geo(
                        decoder_layer,
                        hidden_states,
                        attention_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                        llm_geo_pos,
                        llm_geo_mask,
                    )

                hidden_states = layer_outputs[0]
                if use_cache:
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]
                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

            hidden_states = self.norm(hidden_states)

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            next_cache = None
            if use_cache:
                next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

            if not return_dict:
                outputs = tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
            else:
                outputs = BaseModelOutputWithPast(
                    last_hidden_state=hidden_states,
                    past_key_values=next_cache,
                    hidden_states=all_hidden_states,
                    attentions=all_self_attns,
                )
            self._last_llm_visual_3d_rope_stats = collect_qwen2_visual_3d_rope_stats(self)
            return outputs
        finally:
            clear_qwen2_visual_3d_rope_context(self)


class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config):
        install_qwen2_visual_3d_rope_attention()
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None

        self.model = LlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()
        if getattr(config, "spatial_rank_projection_dim", None) is not None:
            self.initialize_spatial_rank_head(output_dim=int(config.spatial_rank_projection_dim))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        # 创建模型实例
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        # 加载自定义权重
        if model.get_spatial_tower() is not None:
            zero_spatial_features = getattr(model.config, "zero_spatial_features", False)
            if isinstance(zero_spatial_features, str):
                zero_spatial_features = zero_spatial_features.lower() in {"1", "true", "yes", "y", "on"}

            # Vision-only ablation: keep spatial tower wrapper but skip heavy weight loading.
            if not zero_spatial_features:
                model.get_spatial_tower().is_loaded = False
                model.get_spatial_tower().load_model()
                model.get_spatial_tower().is_loaded = True
                model.get_spatial_tower().to(kwargs.get("torch_dtype", torch.float16))

        return model

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        spatial_features: Optional[Dict[str, torch.FloatTensor]] = None,
        geometry_spatial_features: Optional[Dict[str, torch.FloatTensor]] = None,
        point_maps: Optional[torch.FloatTensor] = None,
        geometry_outputs: Optional[Dict[str, torch.FloatTensor]] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        cache_position=None,
        return_visual_metadata: Optional[bool] = False,
        llm_geo_pos: Optional[torch.FloatTensor] = None,
        llm_geo_mask: Optional[torch.BoolTensor] = None,
        llm_geo_debug: Optional[Dict] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        self.get_model()._last_geometry_projection_outputs = None
        self.get_model()._last_geometry_projection_metrics = None
        self._geometry_projection_last_metrics = {}
        metadata_requested = bool(return_visual_metadata)
        spatial_rank_enabled = bool(
            self.training
            and getattr(self.config, "spatial_rank_loss_enable", False)
            and labels is not None
            and inputs_embeds is None
        )
        original_output_hidden_states = output_hidden_states
        if spatial_rank_enabled:
            return_dict = True
        elif metadata_requested:
            output_hidden_states = True
            return_dict = True

        visual_metadata = None
        llm_geo_debug_info = llm_geo_debug
        llm_visual_3d_rope_enabled = _as_bool_config(getattr(self.config, "llm_visual_3d_rope_enable", False), False)
        should_prepare_multimodal = (
            inputs_embeds is None
            and images is not None
            and input_ids is not None
            and input_ids.shape[1] != 1
        )
        if should_prepare_multimodal:
            prepared = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                spatial_features,
                point_maps,
                modalities,
                image_sizes,
                return_visual_metadata=spatial_rank_enabled or metadata_requested,
                return_llm_geo_metadata=llm_visual_3d_rope_enabled,
                geometry_outputs=geometry_outputs,
                geometry_spatial_features=geometry_spatial_features,
            )
            if (spatial_rank_enabled or metadata_requested) and llm_visual_3d_rope_enabled:
                (
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels,
                    visual_metadata,
                    llm_geo_pos,
                    llm_geo_mask,
                    llm_geo_debug_info,
                ) = prepared
            elif spatial_rank_enabled or metadata_requested:
                (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, visual_metadata) = prepared
            elif llm_visual_3d_rope_enabled:
                (
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels,
                    llm_geo_pos,
                    llm_geo_mask,
                    llm_geo_debug_info,
                ) = prepared
            else:
                (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = prepared
        elif llm_visual_3d_rope_enabled and llm_geo_debug_info is None:
            llm_geo_debug_info = {"skip_reason": "text_only_or_cached_decode"}

        if dpo_forward:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                llm_geo_pos=llm_geo_pos,
                llm_geo_mask=llm_geo_mask,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        else:
            h1_holder = {}
            hook_handle = None
            if spatial_rank_enabled:
                first_block = self.model.layers[0]

                def capture_h1(_module, _inputs, output):
                    h1_holder["h1"] = output[0] if isinstance(output, (tuple, list)) else output

                hook_handle = first_block.register_forward_hook(capture_h1)
            try:
                output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
                output_hidden_states = (
                    output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
                )
                return_dict = return_dict if return_dict is not None else self.config.use_return_dict
                model_outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    llm_geo_pos=llm_geo_pos,
                    llm_geo_mask=llm_geo_mask,
                )
                if llm_visual_3d_rope_enabled:
                    current_stats = getattr(self.get_model(), "_last_llm_visual_3d_rope_stats", None)
                    if current_stats:
                        has_active_geo = any(
                            (not item.get("skipped", False)) and int(item.get("num_valid_geo_tokens", 0) or 0) > 0
                            for item in current_stats
                        )
                        if has_active_geo:
                            self._last_llm_visual_3d_rope_prefill_stats = current_stats
                            self._last_llm_geo_prefill_debug = llm_geo_debug_info
                        else:
                            self._last_llm_visual_3d_rope_decode_stats = current_stats
                            self._last_llm_geo_decode_debug = llm_geo_debug_info
                hidden_states = model_outputs[0]
                logits = self.lm_head(hidden_states)
                logits = logits.float()

                loss = None
                if labels is not None:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss_fct = CrossEntropyLoss()
                    shift_logits = shift_logits.view(-1, self.config.vocab_size)
                    shift_labels = shift_labels.view(-1)
                    shift_labels = shift_labels.to(shift_logits.device)
                    loss = loss_fct(shift_logits, shift_labels)

                if not return_dict:
                    output = (logits,) + model_outputs[1:]
                    outputs = (loss,) + output if loss is not None else output
                else:
                    outputs = CausalLMOutputWithPast(
                        loss=loss,
                        logits=logits,
                        past_key_values=model_outputs.past_key_values,
                        hidden_states=model_outputs.hidden_states,
                        attentions=model_outputs.attentions,
                    )
            finally:
                if hook_handle is not None:
                    hook_handle.remove()
            if metadata_requested:
                self._last_visual_metadata = visual_metadata
            if llm_visual_3d_rope_enabled:
                self._last_llm_geo_debug = llm_geo_debug_info
            geometry_projection_outputs = getattr(self.get_model(), "_last_geometry_projection_outputs", None)
            use_geometry_projection = getattr(self.config, "use_geometry_aware_projection", False)
            if isinstance(use_geometry_projection, str):
                use_geometry_projection = use_geometry_projection.lower() in {"1", "true", "yes", "y", "on"}
            use_auxiliary_geometry_head = getattr(self.config, "use_auxiliary_geometry_head", True)
            if isinstance(use_auxiliary_geometry_head, str):
                use_auxiliary_geometry_head = use_auxiliary_geometry_head.lower() in {"1", "true", "yes", "y", "on"}
            use_auxiliary_geometry_loss = getattr(self.config, "use_auxiliary_geometry_loss", True)
            if isinstance(use_auxiliary_geometry_loss, str):
                use_auxiliary_geometry_loss = use_auxiliary_geometry_loss.lower() in {"1", "true", "yes", "y", "on"}
            geometry_loss_enabled = bool(
                self.training
                and labels is not None
                and geometry_projection_outputs is not None
                and geometry_projection_outputs.get("loss_geo") is not None
                and use_geometry_projection
                and use_auxiliary_geometry_head
                and use_auxiliary_geometry_loss
            )
            if not spatial_rank_enabled and not geometry_loss_enabled:
                return outputs

            ce_loss = outputs.loss
            if geometry_loss_enabled and not spatial_rank_enabled:
                loss_geo = geometry_projection_outputs["loss_geo"]
                lambda_geo = float(getattr(self.config, "lambda_geo", 0.1))
                total_loss = ce_loss + lambda_geo * loss_geo
                self._geometry_projection_last_metrics = {
                    "geometry_loss_lm": float(ce_loss.detach().float().item()),
                    "geometry_loss_geo": float(loss_geo.detach().float().item()),
                    "geometry_loss_total": float(total_loss.detach().float().item()),
                    "lambda_geo": lambda_geo,
                }
                return CausalLMOutputWithPast(
                    loss=total_loss,
                    logits=outputs.logits,
                    past_key_values=outputs.past_key_values,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )

            h1 = h1_holder.get("h1", None)
            if h1 is None:
                raise RuntimeError("Spatial ranking loss could not capture H1 from self.model.layers[0].")
            rank_loss, rank_metrics = self.compute_spatial_ranking_loss(
                h1,
                visual_metadata,
                spatial_features,
                debug_checks=bool(getattr(self.config, "spatial_rank_debug_checks", False)),
            )
            lambda_sim = float(getattr(self.config, "lambda_sim", 0.01))
            total_loss = ce_loss + lambda_sim * rank_loss
            if geometry_loss_enabled:
                loss_geo = geometry_projection_outputs["loss_geo"]
                lambda_geo = float(getattr(self.config, "lambda_geo", 0.1))
                total_loss = total_loss + lambda_geo * loss_geo
            self._spatial_rank_last_metrics = dict(rank_metrics)
            self._spatial_rank_last_metrics.update({
                "spatial_rank_ce_loss": float(ce_loss.detach().float().item()),
                "spatial_rank_total_loss": float(total_loss.detach().float().item()),
                "spatial_rank_lambda": lambda_sim,
            })
            if geometry_loss_enabled:
                self._spatial_rank_last_metrics.update({
                    "geometry_loss_geo": float(loss_geo.detach().float().item()),
                    "geometry_loss_weighted": float((loss_geo.detach().float() * lambda_geo).item()),
                    "lambda_geo": lambda_geo,
                })

            return CausalLMOutputWithPast(
                loss=total_loss,
                logits=outputs.logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states if original_output_hidden_states else None,
                attentions=outputs.attentions,
            )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        spatial_features: Optional[torch.Tensor] = None,
        geometry_spatial_features: Optional[torch.Tensor] = None,
        point_maps: Optional[torch.Tensor] = None,
        geometry_outputs: Optional[Dict[str, torch.Tensor]] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        llm_visual_3d_rope_enabled = _as_bool_config(getattr(self.config, "llm_visual_3d_rope_enable", False), False)
        if llm_visual_3d_rope_enabled:
            for attr in (
                "_last_llm_visual_3d_rope_prefill_stats",
                "_last_llm_geo_prefill_debug",
                "_last_llm_visual_3d_rope_decode_stats",
                "_last_llm_geo_decode_debug",
            ):
                if hasattr(self, attr):
                    delattr(self, attr)
        if images is not None:
            prepared = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                spatial_features,
                point_maps,
                modalities,
                image_sizes=image_sizes,
                return_llm_geo_metadata=llm_visual_3d_rope_enabled,
                geometry_outputs=geometry_outputs,
                geometry_spatial_features=geometry_spatial_features,
            )
            if llm_visual_3d_rope_enabled:
                (inputs, position_ids, attention_mask, _, inputs_embeds, _, llm_geo_pos, llm_geo_mask, llm_geo_debug) = prepared
            else:
                (inputs, position_ids, attention_mask, _, inputs_embeds, _) = prepared
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
            llm_geo_pos = None
            llm_geo_mask = None
            llm_geo_debug = None

        try:
            return super().generate(
                position_ids=position_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                llm_geo_pos=llm_geo_pos,
                llm_geo_mask=llm_geo_mask,
                llm_geo_debug=llm_geo_debug,
                **kwargs,
            )
        finally:
            clear_qwen2_visual_3d_rope_context(self.model)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        geometry_outputs = kwargs.pop("geometry_outputs", None)
        geometry_spatial_features = kwargs.pop("geometry_spatial_features", None)
        llm_geo_pos = kwargs.pop("llm_geo_pos", None)
        llm_geo_mask = kwargs.pop("llm_geo_mask", None)
        llm_geo_debug = kwargs.pop("llm_geo_debug", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if geometry_outputs is not None:
            inputs["geometry_outputs"] = geometry_outputs
        if geometry_spatial_features is not None:
            inputs["geometry_spatial_features"] = geometry_spatial_features
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        if llm_geo_pos is not None:
            inputs["llm_geo_pos"] = llm_geo_pos
        if llm_geo_mask is not None:
            inputs["llm_geo_mask"] = llm_geo_mask
        if llm_geo_debug is not None:
            inputs["llm_geo_debug"] = llm_geo_debug
        return inputs


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)
