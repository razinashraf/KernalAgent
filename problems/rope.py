"""Rotary position embedding (RoPE), "rotate half" style as in LLaMA.

Elementwise-ish, but each output element needs TWO input elements (x[d] and x[d + D/2])
plus a cos/sin value that is shared by every batch and head. A good kernel loads the
cos/sin row once and reuses it across heads.
"""

import torch

NAME = "rope"
CATEGORY = "elementwise"
DESCRIPTION = (
    "Rotary position embedding, rotate-half style. x has shape (B, S, H, D) with D even; "
    "cos and sin have shape (S, D/2). With x1 = x[..., :D/2], x2 = x[..., D/2:] and "
    "c = cos[s], s_ = sin[s] for sequence position s: "
    "y[..., :D/2] = x1 * c - x2 * s_ and y[..., D/2:] = x2 * c + x1 * s_. "
    "All inputs are contiguous. Return a new tensor of shape (B, S, H, D) with x's dtype."
)
DTYPES = [torch.float16, torch.float32]
TOLERANCES = {
    torch.float32: {"atol": 1e-5, "rtol": 1e-5},
    torch.float16: {"atol": 1e-2, "rtol": 1e-2},
}
SHAPES = {
    "small": {"B": 1, "S": 512, "H": 8, "D": 64},
    "medium": {"B": 4, "S": 2048, "H": 16, "D": 128},
    "large": {"B": 8, "S": 4096, "H": 32, "D": 128},
}
EDGE_SHAPES = [
    {"B": 1, "S": 1, "H": 1, "D": 2},
    {"B": 1, "S": 1000, "H": 3, "D": 64},
    {"B": 2, "S": 77, "H": 5, "D": 80},   # D/2 = 40: not a power of 2
]


def random_shape(rng):
    return {
        "B": rng.randint(1, 4),
        "S": rng.randint(1, 1024),
        "H": rng.randint(1, 16),
        "D": rng.choice([32, 64, 80, 96, 128]),
    }


def make_inputs(shape, dtype, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    B, S, H, D = shape["B"], shape["S"], shape["H"], shape["D"]
    x = torch.randn(B, S, H, D, device="cuda", dtype=dtype, generator=g)
    # Real RoPE frequencies: position * 10000^(-2i/D)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, D, 2, device="cuda", dtype=torch.float32) / D))
    freqs = torch.outer(torch.arange(S, device="cuda", dtype=torch.float32), inv_freq)
    return [x, freqs.cos().to(dtype), freqs.sin().to(dtype)]


def reference(x, cos, sin):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    c = cos[None, :, None, :]   # broadcast over batch and heads
    s = sin[None, :, None, :]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)
