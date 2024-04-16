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


def all_to_all_4D_async(
    input: torch.tensor,
    output: torch.tensor,
    scatter_idx: int = 2,
    gather_idx: int = 1,
    group=None,
) -> torch.tensor:
    """
    all-to-all for QKV

    Args:
        input (torch.tensor): a tensor sharded along dim scatter dim
        output (torch.tensor): output tensor shared along dim gatter.
        scatter_idx (int): default 1
        gather_idx (int): default 2
        group : torch process group

    Returns:
        torch.tensor: resharded tensor (bs, seqlen/P, hc, hs)
    """
    assert (
        input.dim() == 4
    ), f"input must be 4D tensor, got {input.dim()} and shape {input.shape}"

    seq_world_size = dist.get_world_size(group)

    if scatter_idx == 2 and gather_idx == 1:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen/P, hc, hs) output: (bs, seqlen, hc/P, hs)
        bs, shard_seqlen, hc, hs = input.shape
        seqlen = shard_seqlen * seq_world_size
        shard_hc = hc // seq_world_size

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen/P, hc, hs) -reshape-> (bs, seq_len/P, P, hc/P, hs) -transpose(0,2)-> (P, seq_len/P, bs, hc/P, hs)
        input_t = (
            input.reshape(bs, shard_seqlen, seq_world_size, shard_hc, hs)
            .transpose(0, 2)
            .contiguous()
        )

        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, seq_len/P, bs, hc/P, hs) scatter seqlen -all2all-> (P, seq_len/P, bs, hc/P, hs) scatter head
        dist.all_to_all_single(output, input_t, group=group)

        # if scattering the seq-dim, transpose the heads back to the original dimension
        # (seq_len, bs, hc/P, hs) -reshape-> (bs, seq_len, hc/P, hs)
        output = (
            output.reshape(seqlen, bs, shard_hc, hs)
            .transpose(0, 1)
            .contiguous()
            .reshape(bs, seqlen, shard_hc, hs)
        )

        return output

    elif scatter_idx == 1 and gather_idx == 2:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen, hc/P, hs) output: (bs, seqlen/P, hc, hs)
        bs, seqlen, shard_hc, hs = input.shape
        hc = shard_hc * seq_world_size
        shard_seqlen = seqlen // seq_world_size
        seq_world_size = dist.get_world_size(group)

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen, hc/P, hs) -reshape-> (bs, P, seq_len/P, hc/P, hs) -transpose(0, 3)-> (hc/P, P, seqlen/P, bs, hs) -transpose(0, 1) -> (P, hc/P, seqlen/P, bs, hs)
        input_t = (
            input.reshape(bs, seq_world_size, shard_seqlen, shard_hc, hs)
            .transpose(0, 3)
            .transpose(0, 1)
            .contiguous()
            .reshape(seq_world_size, shard_hc, shard_seqlen, bs, hs)
        )

        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, bs x hc/P, seqlen/P, hs) scatter seqlen -all2all-> (P, bs x seq_len/P, hc/P, hs) scatter head
        dist.all_to_all_single(output, input_t, group=group)

        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = (
            output.reshape(hc, shard_seqlen, bs, hs)
            .transpose(0, 2)
            .contiguous()
            .reshape(bs, shard_seqlen, hc, hs)
        )

        # (hc, seqlen/N, bs, hs) -tranpose(0,2)-> (bs, seqlen/N, hc, hs)

        return output
    else:
        raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")


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
        self.stream = torch.cuda.Stream()

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
            query (Tensor): query input to the layer (bs, seqlen/P, hc, hs)
            key (Tensor): key input to the layer (bs, seqlen/P, hc_kv, hs)
            value (Tensor): value input to the layer (bs, seqlen/P, hc_kv, hs)
            args: other args

        Returns:
            * output (Tensor): context output
        """

        # un*ud = hc

        ulysses_degree = dist.get_world_size(self.ulysses_pg)

        bs, shard_seqlen, hc, hs = query.shape
        bs, shard_seqlen, hc_kv, hs = key.shape
        seq_len = shard_seqlen * ulysses_degree
        un = hc // ulysses_degree
        un_kv = hc_kv // ulysses_degree

        assert un_kv == un, f"un_kv {un_kv} un {un}"

        qkv = torch.cat([query, key, value]).contiguous()
        # (3*bs, seqlen/P, hc, hs) -> (hc, seqlen/P, 3*bs, hs) -> (un, ud, seqlen/P, 3*bs, hs), where hc = un*ud
        qkv_list = torch.unbind(
            qkv.transpose(0, 2)
            .contiguous()
            .reshape(un, ulysses_degree, shard_seqlen, 3 * bs, hs)
        )
        # 3xall-to-all output buffer
        qkv_trans_list = [
            torch.zeros(
                ulysses_degree,
                1,
                shard_seqlen,
                3 * bs,
                hs,
                dtype=query.dtype,
                device=query.device,
            )
            for i in range(len(qkv_list))
        ]
        # last all-to-all buffter
        context_layer_list = [
            torch.zeros(
                ulysses_degree,
                1,
                shard_seqlen,
                bs,
                hs,
                dtype=query.dtype,
                device=query.device,
            )
            for i in range(len(qkv_list))
        ]

        comm_handle_list = []

        # un * (ud, shard_seqlen, 3*bs, hs)
        for i, qkv in enumerate(qkv_list):
            # with torch.cuda.stream(self.stream):
            ret = dist.all_to_all_single(
                qkv_trans_list[i], qkv, group=self.ulysses_pg, async_op=True
            )
            comm_handle_list.append(ret)

        last_comm_handle_list = []
        for i, qkv_trans in enumerate(qkv_trans_list):
            comm_handle_list[i].wait()
            qkv_trans = (
                qkv_trans.reshape(seq_len, 3 * bs, 1, hs)
                .transpose(0, 1)
                .contiguous()
                .reshape(3 * bs, seq_len, 1, hs)
            )

            # qkv_trans = all_to_all_4D_async(qkv, qkv_trans_list[i], self.scatter_idx, self.gather_idx, self.ulysses_pg)
            qkv_trans = torch.chunk(qkv_trans, 3, dim=0)

            out = self.ring_attn_fn(
                qkv_trans[0],
                qkv_trans[1],
                qkv_trans[2],
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

            context_layer = (
                context_layer.reshape(bs, ulysses_degree, shard_seqlen, 1, hs)
                .transpose(0, 3)
                .transpose(0, 1)
                .contiguous()
                .reshape(ulysses_degree, 1, shard_seqlen, bs, hs)
            )
            # with torch.cuda.stream(self.stream):
            ret = dist.all_to_all_single(
                context_layer_list[i],
                context_layer,
                group=self.ulysses_pg,
                async_op=True,
            )
            last_comm_handle_list.append(ret)
            # output = SeqAllToAll4D.apply(
            #     self.ulysses_pg, context_layer, self.gather_idx, self.scatter_idx
            # )
            # print(f"q.shape {qkv[0].shape}, context_layer {context_layer.shape}")
            # out (nu, bs, seqlen, 1, hs)
            # context_layer_list.append(output)

        # hc = un * P
        # un x (hc = P, seq_len/P, bs, hs) -> (bs, seq_len, hc = P, hs)
        for i, ret in enumerate(last_comm_handle_list):
            ret.wait()
            context_layer_list[i] = (
                context_layer_list[i]
                .reshape(ulysses_degree, shard_seqlen, bs, hs)
                .transpose(0, 2)
                .contiguous()
                .reshape(bs, shard_seqlen, ulysses_degree, hs)
            )

        output = torch.cat(context_layer_list, dim=2)
        return output

    def backward(self, *args, **kwargs):
        raise RuntimeError(
            "Backward computation is not allowed for AsyncLongContextAttention."
        )
