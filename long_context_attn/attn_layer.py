from ds_ulysses_attn.utils import _SeqAllToAll4D
from ring_flash_attn import (
    ring_flash_attn_func,
)
import torch

from typing import Any
from torch import Tensor

import torch.distributed as dist


class LongContextAttention(torch.nn.Module):
    """Initialization.

    Arguments:
        ulysses_pg (ProcessGroup): ulysses process group
        ring_pg (ProcessGroup): ring process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
    """

    def __init__(
        self,
        ulysses_pg: dist.ProcessGroup,
        ring_pg: dist.ProcessGroup,
        scatter_idx: int = 2,
        gather_idx: int = 1,
    ) -> None:

        super(LongContextAttention, self).__init__()

        self.ring_pg = ring_pg
        self.ulysses_pg = ulysses_pg
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        *args: Any,
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

        print(
            f"before all2all query.shape: {query.shape}, key.shape: {key.shape}, value.shape: {value.shape}"
        )

        # scatter 2, gather 1
        query_layer = _SeqAllToAll4D.apply(
            self.ulysses_pg, query, self.scatter_idx, self.gather_idx
        )
        key_layer = _SeqAllToAll4D.apply(
            self.ulysses_pg, key, self.scatter_idx, self.gather_idx
        )
        value_layer = _SeqAllToAll4D.apply(
            self.ulysses_pg, value, self.scatter_idx, self.gather_idx
        )

        # out shape : e.g., [s:h/p:]
        print(
            f"after all2all query_layer.shape: {query_layer.shape}, key_layer.shape: {key_layer.shape}, value_layer.shape: {value_layer.shape}"
        )

        context_layer, _, _ = ring_flash_attn_func(
            query_layer,
            key_layer,
            value_layer,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=return_attn_probs,
            group=self.ring_pg,
        )

        print(f"context_layer shape {context_layer.shape}")

        # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
        # scatter 1, gather 2
        output = _SeqAllToAll4D.apply(
            self.ulysses_pg, context_layer, self.gather_idx, self.scatter_idx
        )

        print(f"output shape {output.shape}")
        # out e.g., [s/p::h]
        return output
