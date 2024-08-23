# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from typing import Any
from torch import Tensor

import torch.distributed as dist
from flash_attn import flash_attn_func
from yunchang.comm.all_to_all import SeqAllToAll4D
import torch.nn.functional as F

def torch_attn(query,
            key,
            value,
            dropout_p=0.0, 
            softmax_scale=None, 
            causal=False,
            window_size=(-1, -1), alibi_slopes=None, deterministic=False,
            return_attn_probs=False,
            ):
    batch_size, seq_len, hs, hd = query.size()
    query = query.view(batch_size, -1, hs, hd).transpose(1, 2)
    key = key.view(batch_size, -1, hs, hd).transpose(1, 2)
    value = value.view(batch_size, -1, hs, hd).transpose(1, 2)

    # the output of sdp = (batch, num_heads, seq_len, head_dim)
    # TODO: add support for attn.scale when we move to Torch 2.1
    hidden_states = F.scaled_dot_product_attention(
        query, key, value, dropout_p=dropout_p, is_causal=causal
    )

    hidden_states = hidden_states.transpose(1, 2).reshape(
        batch_size, -1, hs, hd
    )
    hidden_states = hidden_states.to(query.dtype)
    return hidden_states

class UlyssesAttention(torch.nn.Module):
    """Initialization.

    Arguments:
        local_attention (Module): local attention with q,k,v
        sequence_process_group (ProcessGroup): sequence parallel process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
    """

    def __init__(
        self,
        sequence_process_group: dist.ProcessGroup = None,
        scatter_idx: int = 2,
        gather_idx: int = 1,
        use_fa : bool = True 
    ) -> None:

        super(UlyssesAttention, self).__init__()
        self.spg = sequence_process_group
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        self.use_fa = use_fa
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpu_name = torch.cuda.get_device_name(device)
        if "Turing" in gpu_name or "Tesla" in gpu_name or "T4" in gpu_name:
            self.use_fa = False

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        *args: Any
    ) -> Tensor:
        """forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """
        # TODO Merge three alltoall calls into one
        # TODO (Reza): change the api on the megatron-deepspeed side so that we only receive all data (q,k, and v) together!
        # in shape : e.g.,  [s/p:h:]
        # (bs, seq_len/N, head_cnt, head_size) -> (bs, seq_len, head_cnt/N, head_size)

        # scatter 2, gather 1
        q = SeqAllToAll4D.apply(self.spg, query, self.scatter_idx, self.gather_idx)
        k = SeqAllToAll4D.apply(self.spg, key, self.scatter_idx, self.gather_idx)
        v = SeqAllToAll4D.apply(self.spg, value, self.scatter_idx, self.gather_idx)

        if self.use_fa:
            fn = flash_attn_func
        else:
            fn = torch_attn
            
        context_layer = fn(
            q,
            k,
            v,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=return_attn_probs,
        )

        if isinstance(context_layer, tuple):
            context_layer = context_layer[0]

        # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
        # scatter 1, gather 2
        output = SeqAllToAll4D.apply(
            self.spg, context_layer, self.gather_idx, self.scatter_idx
        )

        # out e.g., [s/p::h]
        return output

