# ------------------------------------------------------------------------
# Modified from MoE-LLaVA(https://github.com/PKU-YuanGroup/MoE-LLaVA)
# ------------------------------------------------------------------------

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import CausalLMOutputWithPast

from dataclasses import dataclass
from einops import rearrange, repeat, reduce, pack, unpack

from transformers.utils import ModelOutput
from transformers.activations import ACT2FN


def MixtralDecoderLayerMOEBlock_forward(self):
    def forward(hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        router_z_loss = torch.logsumexp(router_logits, dim = -1)
        router_z_loss = torch.square(router_z_loss)            
        router_z_loss = router_z_loss.mean()

        routing_weights = nn.functional.softmax(router_logits, dim=1, dtype=torch.float)

        density_1_proxy = reduce(routing_weights, '... n e -> ... e', 'mean')

        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

        one_hot_gate_indices = nn.functional.one_hot(rearrange(selected_experts, '... k -> k ...'), self.num_experts).float()[0]
        density_1 = reduce(one_hot_gate_indices, '... n e -> ... e', 'mean')
        balance_loss = (density_1_proxy * density_1).mean() * float(self.num_experts ** 2)

        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        return final_hidden_states, (balance_loss, router_z_loss)
    return forward

@dataclass
class SMoECausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
