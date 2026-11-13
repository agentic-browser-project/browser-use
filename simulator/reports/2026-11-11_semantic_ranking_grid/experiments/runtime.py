"""Pinned BrowSA parser, packing and selector with a small diagnostic adapter."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
import types

import numpy as np
import torch

ROOT = Path('/workspace/semantic_ranking_grid_20261111')
SOURCE = ROOT / 'source'
PREVIOUS = Path('/workspace/anchor_alpha_grid_20261110')
CONFIGS = [(a, alpha) for a in (1, 4, 8, 16, 32) for alpha in (.125, .25, .375, .5)]
DIAGNOSTIC_LAYERS = (0, 16, 32, 48)

# Import numerical reference modules without the package's SGLang registration.
# Their source remains byte-for-byte unchanged.
package = types.ModuleType('bsa_sglang')
package.__path__ = [str(SOURCE / 'bsa_sglang')]
sys.modules.setdefault('bsa_sglang', package)
sys.path.insert(0, str(SOURCE / 'python'))
from tree_parser import extract_semantic_chunks_v2
from bsa_sglang.boundary_packing import middle_chunk_bounds
from bsa_sglang.packing_gpu import verify_layout


@lru_cache(None)
def extension(kind):
    from torch.utils.cpp_extension import load
    files = {'select': ('selection_tuned.cu', '--use_fast_math'),
             'packing': ('packing_cuda.cu', '--use_fast_math'),
             'linkage': ('packing_linkage_cuda.cu', '--fmad=false'),
             'qmean': ('score_post.cu', '--fmad=false')}
    filename, flag = files[kind]
    path = SOURCE / 'bsa_sglang' / filename
    suffix = hashlib.sha256(path.read_bytes()).hexdigest()[:10]
    return load(name=f'browsa_ranking_{kind}_{suffix}', sources=[str(path)],
                extra_cuda_cflags=['-O3', flag], verbose=False)


def chunk_spans(ids, tokenizer, cap):
    texts = {token: tokenizer.decode([token]) for token in set(ids)}
    chunks = extract_semantic_chunks_v2([texts[t] for t in ids], min_chunk_size=6, max_chunk_size=cap)
    spans = np.array([(c.start_idx, c.end_idx) for c in chunks], dtype=np.int64)
    assert spans[0, 0] == 0 and spans[-1, 1] == len(ids) - 1
    assert np.array_equal(spans[1:, 0], spans[:-1, 1] + 1)
    lengths = spans[:, 1] - spans[:, 0] + 1
    assert (lengths > 0).all() and (lengths <= cap).all()
    return spans


def envelopes(keys, spans):
    """BF16 extrema and transforms, retaining serving's rounding boundaries."""
    lengths = spans[:, 1] - spans[:, 0] + 1
    offsets = np.arange(int(lengths.max()))
    idx = torch.as_tensor(np.minimum(spans[:, 0, None] + offsets, len(keys)-1), device=keys.device)
    valid = torch.as_tensor(offsets[None] < lengths[:, None], device=keys.device)
    values = keys[idx]
    mx = values.masked_fill(~valid[:, :, None, None], -torch.inf).amax(1)
    mn = values.masked_fill(~valid[:, :, None, None], torch.inf).amin(1)
    center = ((mx + mn) * .5).permute(1, 0, 2).contiguous()
    width = ((mx - mn) * .5).permute(1, 0, 2).contiguous()
    return mx, mn, center, width


def components(query, center, width):
    """Queries [Q,64,128], common static summaries [8,C,128]."""
    qg, qa = extension('qmean').qmean(query.contiguous(), 8, 8)
    count = len(query)
    c = torch.bmm(center[None].expand(count, -1, -1, -1).reshape(count*8, -1, 128),
                  qg.reshape(count*8, 128, 1)).reshape(count, 8, -1)
    w = torch.bmm(width[None].expand(count, -1, -1, -1).reshape(count*8, -1, 128),
                  qa.reshape(count*8, 128, 1)).reshape(count, 8, -1)
    return c, w


def make_layout(mx, mn, lengths, page_size, anchor, alpha, verify=False):
    """Return page ranges and physical-to-logical prompt token permutation."""
    lo, hi = middle_chunk_bounds(lengths, page_size, 2*page_size, False)
    offsets = np.r_[0, np.cumsum(lengths)]
    plen = int(offsets[-1])
    factor = torch.as_tensor((anchor / lengths[lo:hi]) ** alpha,
                             dtype=torch.float32, device=mx.device)
    distance = extension('packing').distances(mx[None, lo:hi], mn[None, lo:hi], factor)
    first, last, middle = extension('linkage').layout(
        distance, torch.as_tensor(offsets, dtype=torch.int32, device=mx.device),
        lo, hi, int(offsets[hi] - offsets[lo]), page_size)
    if verify:
        verify_layout(distance, lengths, lo, hi, page_size, first, last, middle)
    permutation = torch.arange(plen, device=mx.device, dtype=torch.int32)[None].repeat(8, 1)
    permutation[:, offsets[lo]:offsets[hi]] = middle[0] + int(offsets[lo])
    return first[0], last[0], permutation


def natural_layout(spans, heads=8, device='cuda', page_size=32):
    first = torch.as_tensor(spans[:, 0] // page_size, dtype=torch.int32, device=device)[None].repeat(heads, 1)
    last = torch.as_tensor(spans[:, 1] // page_size, dtype=torch.int32, device=device)[None].repeat(heads, 1)
    perm = torch.arange(int(spans[-1, 1])+1, dtype=torch.int32, device=device)[None].repeat(heads, 1)
    return first, last, perm


def select_pages(c, w, lengths, factor, first, last, seq, plen, page_size, ratio=.125):
    """Use the actual distinct-page admission CUDA kernel, including reserves."""
    batch, heads, count = c.shape
    device = c.device
    seq = torch.as_tensor(seq, dtype=torch.int32, device=device).expand(batch).contiguous()
    total_pages = int((int(seq.max()) + page_size-1) // page_size)
    lens = torch.as_tensor(lengths, dtype=torch.float32, device=device)[None].expand(batch, -1).contiguous()
    factor = torch.as_tensor(factor, dtype=torch.float32, device=device)
    if factor.ndim == 1:
        factor = factor[None].expand(batch, -1).contiguous()
    def expand_layout(value):
        return value[None].expand(batch, -1, -1).contiguous() if value.ndim == 2 else value.contiguous()
    first, last = expand_layout(first), expand_layout(last)
    budget = torch.floor(((seq + page_size - 1) // page_size).float() * ratio)
    aif = torch.full((batch,), plen, dtype=torch.int32, device=device)
    phys = torch.arange(total_pages, dtype=torch.int32, device=device)[None].expand(batch, -1).contiguous()
    ends = torch.as_tensor(np.cumsum(lengths), dtype=torch.int32, device=device)[None].expand(batch, -1).contiguous()
    local, counts = extension('select').select(c.contiguous(), w.contiguous(), lens, factor, budget,
        first, last, seq, aif, phys, page_size, 256, False, ends,
        2*page_size, page_size, distinct_page_budget=True)
    return local.reshape(batch, heads, total_pages), counts.reshape(batch, heads), budget


def selected_token_mask(local, counts, permutation, seq, plen, page_size, total_tokens):
    """Convert fetched physical pages to original token coordinates."""
    batch, heads, pages = local.shape
    device = local.device
    page_mask = torch.zeros((batch, heads, pages+1), dtype=torch.bool, device=device)
    valid = torch.arange(pages, device=device)[None, None] < counts[:, :, None]
    indices = torch.where(valid, local.long(), pages)
    page_mask.scatter_(2, indices, True)
    page_mask = page_mask[:, :, :pages]
    # Each original prompt token maps to one physical position. Generated tokens
    # keep their original absolute positions beyond the packed prompt.
    inverse = permutation.long().argsort(dim=-1)
    if inverse.ndim == 2:
        inverse = inverse[None].expand(batch, -1, -1)
    position = torch.arange(total_tokens, device=device)[None, None].expand(batch, heads, -1).clone()
    position[:, :, :plen] = inverse
    positions = (position // page_size).clamp(max=pages-1)
    mask = page_mask.gather(2, positions)
    seq = torch.as_tensor(seq, dtype=torch.int32, device=device).expand(batch)
    mask &= torch.arange(total_tokens, device=device)[None, None] < seq[:, None, None]
    return mask


def cpu_select(scores, spans, first, last, seq, plen, page_size, ratio=.125):
    """Independent page-union reference, stable original-index tie breaking."""
    total = (seq + page_size-1)//page_size
    cap = int(np.floor(total*ratio))
    chosen = set(range(plen//page_size, total))
    for i, (start, end) in enumerate(spans):
        if start < page_size or end + 1 > seq - 2*page_size:
            chosen.update(range(int(first[i]), int(last[i])+1))
    for i in np.argsort(-np.asarray(scores), kind='stable'):
        pages = set(range(int(first[i]), int(last[i])+1))
        if len(chosen | pages) <= cap:
            chosen |= pages
    return sorted(chosen)


if __name__ == '__main__':
    for name in ('packing', 'linkage', 'qmean', 'select'):
        extension(name)
        print('Built', name, flush=True)
