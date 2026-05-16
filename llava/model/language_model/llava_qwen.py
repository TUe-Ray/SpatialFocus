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

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

# from ...constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM

# from .qwen.modeling_qwen import QWenLMHeadModel, QWenModel
# from .qwen.configuration_qwen import QWenConfig


class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenModel, self).__init__(config)


class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config):
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
        if inputs_embeds is None:
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
                geometry_outputs=geometry_outputs,
                geometry_spatial_features=geometry_spatial_features,
            )
            if spatial_rank_enabled or metadata_requested:
                (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, visual_metadata) = prepared
            else:
                (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = prepared

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
                outputs = super().forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            finally:
                if hook_handle is not None:
                    hook_handle.remove()
            if metadata_requested:
                self._last_visual_metadata = visual_metadata
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

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(
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
                geometry_outputs=geometry_outputs,
                geometry_spatial_features=geometry_spatial_features,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        geometry_outputs = kwargs.pop("geometry_outputs", None)
        geometry_spatial_features = kwargs.pop("geometry_spatial_features", None)
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
        return inputs


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)
