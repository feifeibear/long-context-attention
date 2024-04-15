from yunchang.comm.all_to_all import SeqAllToAll4D, SeqAllToAll5D

import torch

from typing import Any
from torch import Tensor

import torch.distributed as dist
from .utils import RING_IMPL_DICT, RING_IMPL_QKVPACKED_DICT
from yunchang.globals import PROCESS_GROUP


def _chunk(t, ulysses_degree):
    bs, local_seqlen, hc, hs = t.shape

    assert hc % ulysses_degree == 0

    un = hc // ulysses_degree

    # (bs, local_seqlen, un, ulysses_degree, hs) -> (un, bs, local_seqlne, ulysses_degree, hs)
    t_list = torch.unbind(
        t.reshape(bs, local_seqlen, un, ulysses_degree, hs)
        .transpose(0, 2)
        .transpose(1, 2)
    )
    return t_list


class AsyncLongContextAttention(torch.nn.Module):
    """Initialization.

    Arguments:
        ulysses_pg (ProcessGroup): ulysses process group
        ring_pg (ProcessGroup): ring process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
    """

    def __init__(
        self,
        scatter_idx: int = 2,
        gather_idx: int = 1,
        ring_impl_type: str = "basic",
    ) -> None:

        super(AsyncLongContextAttention, self).__init__()
        self.ring_pg = PROCESS_GROUP.RING_PG
        self.ulysses_pg = PROCESS_GROUP.ULYSSES_PG

        assert (
            self.ulysses_pg is not None or self.ring_pg is not None
        ), f"use set_seq_parallel_pg() first. Now ulysses pg {self.ulysses_pg} and ring pg {self.ring_pg}"
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        self.ring_attn_fn = RING_IMPL_DICT[ring_impl_type]

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

        # 3 X (bs, seq_len/N, head_cnt, head_size) -> 3 X (bs, seq_len, head_cnt/N, head_size)
        # scatter 2, gather 1
        # (3*bs, seq_len/N, head_cnt, head_size)

        ulysses_degree = dist.get_world_size(self.ulysses_pg)

        # TODO() key hc not the same
        query_list = _chunk(query, ulysses_degree)
        key_list = _chunk(key, ulysses_degree)
        value_list = _chunk(value, ulysses_degree)

        # print(f"query_list len {len(query_list)}")
        context_layer_list = []
        for q, k, v in zip(query_list, key_list, value_list):
            # (3*bs, seq_len, head_cnt/N, head_size)
            # print(f"q shape before attn {q.shape}")
            qkv = torch.cat([q, k, v])
            qkv = SeqAllToAll4D.apply(
                self.ulysses_pg, qkv, self.scatter_idx, self.gather_idx
            )
            qkv = torch.chunk(qkv, 3, dim=0)
            out = self.ring_attn_fn(
                qkv[0],
                qkv[1],
                qkv[2],
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=return_attn_probs,
                group=self.ring_pg,
            )

            if type(out) == tuple:
                context_layer, _, _ = out
            else:
                context_layer = out

            # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
            # scatter 1, gather 2
            output = SeqAllToAll4D.apply(
                self.ulysses_pg, context_layer, self.gather_idx, self.scatter_idx
            )
            # print(f"q.shape {qkv[0].shape}, context_layer {context_layer.shape}")
            # out (nu, bs, seqlen, 1, hs)
            context_layer_list.append(output)

        context_layer = torch.cat(context_layer_list, dim=2)

        # print(f"final context_layer {context_layer.shape}")

        output = context_layer

        # out e.g., [s/p::h]
        return output
