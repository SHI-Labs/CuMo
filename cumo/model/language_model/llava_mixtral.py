#    Copyright 2023 Haotian Liu
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
# ------------------------------------------------------------------------
# Modified from LLaVA (https://github.com/haotian-liu/LLaVA) and MoE-LLaVA(https://github.com/PKU-YuanGroup/MoE-LLaVA)
# Copyright 2024 Jiachen Li
# ------------------------------------------------------------------------

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         MixtralConfig, MixtralModel, MixtralForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

from .smoe_mixtral_helper import SMoECausalLMOutputWithPast, MixtralDecoderLayerMOEBlock_forward


class LlavaMixtralConfig(MixtralConfig):
    model_type = "llava_mixtral"


class LlavaMixtralModel(LlavaMetaModel, MixtralModel):
    config_class = LlavaMixtralConfig

    def __init__(self, config: MixtralConfig):
        super(LlavaMixtralModel, self).__init__(config)


class LlavaMixtralForCausalLM(MixtralForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaMixtralConfig

    def __init__(self, config):
        super(MixtralForCausalLM, self).__init__(config)
        self.model = LlavaMixtralModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

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
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                clip_balance_loss,
                clip_router_z_loss,
                mlp_balance_loss,
                mlp_router_z_loss
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        output_router_logits = True
        ### We set output_router_logits to True and squeeze bzloss into outputs.router_logits. This hack implementation needs to be fixed
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None

        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        b_loss = None
        z_loss = None
        
        if self.config.training:
            if self.config.mlp_smoe or self.config.clip_smoe:
                if self.config.local_rank == 0:
                    print('language loss: ', loss.item())
                if self.config.mlp_smoe:
                    mlp_balance_loss = mlp_balance_loss.sum(dim=-1).mean()
                    mlp_balance_loss = self.config.balance_loss_coef * mlp_balance_loss
                    loss += mlp_balance_loss
                    mlp_router_z_loss = mlp_router_z_loss.sum(dim=-1).mean()
                    mlp_router_z_loss = self.config.router_z_loss_coef * mlp_router_z_loss
                    loss += mlp_router_z_loss
                    if self.config.local_rank == 0:
                        print('mlp balance loss: ', mlp_balance_loss.item(), 'mlp router z loss: ', mlp_router_z_loss.item())
                if self.config.clip_smoe:
                    clip_balance_loss = clip_balance_loss.sum(dim=-1).mean()
                    clip_balance_loss = self.config.balance_loss_coef * clip_balance_loss
                    loss += clip_balance_loss
                    clip_router_z_loss = clip_router_z_loss.sum(dim=-1).mean()
                    clip_router_z_loss = self.config.router_z_loss_coef * clip_router_z_loss
                    loss += clip_router_z_loss
                    if self.config.local_rank == 0:
                        print('clip balance loss: ', clip_balance_loss.item(), 'clip router z loss: ', clip_router_z_loss.item())
        
                balance_loss = [loss_pair[0] for loss_pair in outputs.router_logits]
                b_loss = sum(balance_loss) / len(balance_loss)
                b_loss = self.config.balance_loss_coef * b_loss
                loss += b_loss
                router_z_loss = [loss_pair[1] for loss_pair in outputs.router_logits]
                z_loss = sum(router_z_loss) / len(balance_loss)
                z_loss = self.config.router_z_loss_coef * z_loss
                loss += z_loss
                if self.config.local_rank == 0:
                    print('llm balance loss: ', b_loss.item(), 'llm router z loss: ', z_loss.item())     

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return SMoECausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def initialize_smoe_modules(self, model_args):
        for m in self.model.layers:
            m.block_sparse_moe.forward = MixtralDecoderLayerMOEBlock_forward(m.block_sparse_moe)

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                _,
                _,
                _,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_mixtral", LlavaMixtralConfig)
AutoModelForCausalLM.register(LlavaMixtralConfig, LlavaMixtralForCausalLM)
