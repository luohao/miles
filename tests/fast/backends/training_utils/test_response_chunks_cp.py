"""Zigzag-CP response selection must match the per-span slices it replaced, rows and gradients.

The selection gathers both CP halves with one index_select so backward allocates
one [local_sequence, vocab] gradient per sample instead of one per slice.
"""

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

import miles.backends.training_utils.data.context_parallel as cp_utils
import miles.backends.training_utils.loss.hub.logit_processors as logit_processors
from miles.backends.training_utils.data.context_parallel import get_logits_and_tokens_offset_with_cp


def _reference_rows(total_lengths, response_lengths, cp_rank, cp_size, qkv_format, max_seq_lens):
    """Local logit rows per sample, computed with the original nested slicing."""
    rows, end = [], 0
    for i, (total_length, response_length) in enumerate(zip(total_lengths, response_lengths, strict=True)):
        chunk_size, chunks, logits_offset, _ = get_logits_and_tokens_offset_with_cp(
            total_length,
            response_length,
            qkv_format,
            max_seq_len=max_seq_lens[i] if max_seq_lens is not None else None,
            cp_rank=cp_rank,
            cp_size=cp_size,
        )
        local = torch.arange(end, end + 2 * chunk_size)
        first = local[:chunk_size][logits_offset[0][0] - chunks[0][0] : logits_offset[0][1] - chunks[0][0]]
        second = local[chunk_size:][logits_offset[1][0] - chunks[1][0] : logits_offset[1][1] - chunks[1][0]]
        rows.append(torch.cat([first, second]).tolist())
        end += 2 * chunk_size
    return rows


@pytest.mark.parametrize("cp_size", [2, 4, 8])
@pytest.mark.parametrize("qkv_format", ["thd", "bshd"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "total_lengths,response_lengths",
    [([9, 16], [4, 1]), ([12], [0]), ([7, 5, 20], [6, 2, 13])],
)
def test_one_index_select_matches_the_two_slices(
    monkeypatch, cp_size, qkv_format, dtype, total_lengths, response_lengths
):
    for cp_rank in range(cp_size):
        _check_response_selection(monkeypatch, cp_rank, cp_size, qkv_format, dtype, total_lengths, response_lengths)


def _check_response_selection(monkeypatch, cp_rank, cp_size, qkv_format, dtype, total_lengths, response_lengths):
    state = SimpleNamespace(cp=SimpleNamespace(rank=cp_rank, size=cp_size))
    monkeypatch.setattr(logit_processors, "get_parallel_state", lambda: state)
    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda: state)

    padded_length = max((total + 2 * cp_size - 1) // (2 * cp_size) * (2 * cp_size) for total in total_lengths)
    max_seq_lens = [padded_length] * len(total_lengths) if qkv_format == "bshd" else None
    local_rows = (
        len(total_lengths) * padded_length // cp_size
        if max_seq_lens is not None
        else sum(2 * ((total + 2 * cp_size - 1) // (2 * cp_size)) for total in total_lengths)
    )
    logits = torch.randn(1, local_rows, 5, dtype=dtype, requires_grad=True)
    tokens = [torch.arange(total) for total in total_lengths]
    args = Namespace(qkv_format=qkv_format, true_on_policy_mode=False, allgather_cp=False)

    chunks = list(
        logit_processors._iter_response_chunks(
            logits,
            args=args,
            unconcat_tokens=tokens,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
            include_response_indices=True,
        )
    )

    flat = logits.detach().squeeze(0)
    for (logits_chunk, tokens_chunk, response_indices), rows in zip(
        chunks,
        _reference_rows(total_lengths, response_lengths, cp_rank, cp_size, qkv_format, max_seq_lens),
        strict=True,
    ):
        assert torch.equal(logits_chunk.detach(), flat[rows])
        assert logits_chunk.size(0) == tokens_chunk.size(0) == len(response_indices)
    for (_, tokens_chunk, response_indices), tokens_i, total_length, response_length in zip(
        chunks, tokens, total_lengths, response_lengths, strict=True
    ):
        assert torch.equal(
            tokens_chunk, tokens_i[[total_length - response_length + index for index in response_indices]]
        )

    sum(chunk.sum() for chunk, _, _ in chunks).backward()
    expected_grad = torch.zeros_like(flat)
    for rows in _reference_rows(total_lengths, response_lengths, cp_rank, cp_size, qkv_format, max_seq_lens):
        expected_grad[rows] += 1
    assert torch.equal(logits.grad.squeeze(0), expected_grad)


def test_inconsistent_cp_padding_fails_before_index_select(monkeypatch):
    state = SimpleNamespace(cp=SimpleNamespace(rank=0, size=2))
    monkeypatch.setattr(logit_processors, "get_parallel_state", lambda: state)
    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda: state)
    args = Namespace(qkv_format="thd", true_on_policy_mode=False, allgather_cp=False)

    with pytest.raises(AssertionError, match="sample 0: local logits have 7 rows.*requires at least 8"):
        list(
            logit_processors._iter_response_chunks(
                torch.randn(1, 7, 5),
                args=args,
                unconcat_tokens=[torch.arange(16)],
                total_lengths=[16],
                response_lengths=[15],
                include_response_indices=True,
            )
        )
