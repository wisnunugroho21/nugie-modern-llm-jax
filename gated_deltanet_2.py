"""
Gated Delta Rule-2 — naive / pedagogical implementations
Paper: https://arxiv.org/abs/2605.22791  (Eq. 9-10)

Two equivalent formulations of:

    S_bar_t = Diag(alpha_t) S_{t-1}                     -- per-channel decay
    r_t     = S_bar_t^T e_t,  e_t = b_t o k_t           -- read along erase dir
    S_t     = S_bar_t + k_t (z_t - r_t)^T,  z_t = w_t o v_t  -- write
    o_t     = S_t^T q_t                                  -- output

Equivalently (compact matrix form, Eq. 10):
    S_t = (I - k_t (b_t o k_t)^T) Diag(alpha_t) S_{t-1} + k_t (w_t o v_t)^T

Argument convention (both functions identical):
  beta   = b_t  in [0,1]^{d_k}  erase gate
  gamma  = w_t  in [0,1]^{d_v}  write gate                   [C1]
  delta  = a_t  in (0,1]^{d_k}  per-channel decay            [C1]

[C1] Paper uses gamma for cumulative log-decay and 'w' for the write gate.
     Consider renaming: beta->b, gamma->w, delta->alpha for paper alignment.

Fixes vs. original:
  [F1] Wrong shape comments on chunk slices fixed
       (was "(B, 1, dim)", now correctly "(B, C, 1, dim)").
  [F2] `sum` renamed to `accum` -- shadows Python built-in.
  [F3] Added divisibility assertion to catch silent token loss.
  [F4] Removed dead `S_c = S_t` init; final state tracked via `S_last`.
  [F5] Inner-loop shape comments corrected to (B, 1, d_k/d_v).
"""

import jax
import jax.numpy as jnp
from flax import nnx

# ---------------------------------------------------------------------------
# Token-by-token sequential reference
# ---------------------------------------------------------------------------


def sequential_forward(
    query: jax.Array,  # (B, L, d_k)
    key: jax.Array,  # (B, L, d_k)
    value: jax.Array,  # (B, L, d_v)
    beta: jax.Array,  # (B, L, d_k)  erase gate b_t
    gamma: jax.Array,  # (B, L, d_v)  write gate w_t  [C1]
    delta: jax.Array,  # (B, L, d_k)  per-channel decay alpha_t  [C1]
) -> jax.Array:  # (B, L, d_v)
    """
    Token-by-token implementation of Gated Delta Rule-2 (paper Eq. 10).
    O(L * d_k^2 * d_v) -- use only for verification / learning.
    """
    batch_size, seq_len, query_dim = query.shape
    value_dim = value.shape[-1]

    S_t = jnp.zeros((batch_size, query_dim, value_dim), dtype=query.dtype)
    I = jnp.eye(query_dim, dtype=query.dtype)  # (d_k, d_k); broadcasts over B

    outputs: list[jax.Array] = []

    for t in range(seq_len):
        # Slice token t and add a unit "seq" axis for matmul compatibility.
        q_t = jnp.expand_dims(query[:, t, :], axis=1)  # (B, 1, d_k)
        k_t = jnp.expand_dims(key[:, t, :], axis=1)  # (B, 1, d_k)
        v_t = jnp.expand_dims(value[:, t, :], axis=1)  # (B, 1, d_v)
        b_t = jnp.expand_dims(beta[:, t, :], axis=1)  # (B, 1, d_k)
        w_t = jnp.expand_dims(gamma[:, t, :], axis=1)  # (B, 1, d_v)
        d_t = jnp.expand_dims(delta[:, t, :], axis=1)  # (B, 1, d_k)

        # ── Transition matrix  A_t = (I - k_t e_t^T) Diag(alpha_t) ─────
        #   k_t.swapaxes(1,2)  : (B, d_k, 1)   column vector
        #   b_t * k_t          : (B, 1,  d_k)   row vector e_t^T = (b_t o k_t)^T
        #   outer product      : (B, d_k, d_k)  k_t e_t^T
        #   * d_t (broadcast)  : (B, 1,  d_k) -> column-wise scale by alpha_t
        #   A_t[i,j] = (delta_{ij} - k_i * e_j) * alpha_j          (Eq. 10)
        A_t = (I - k_t.swapaxes(1, 2) * (b_t * k_t)) * d_t  # (B, d_k, d_k)

        # ── Write term  B_t = k_t z_t^T,  z_t = w_t o v_t ─────────────
        #   k_t.swapaxes(1,2) : (B, d_k, 1)
        #   w_t * v_t         : (B,  1, d_v)
        #   outer product     : (B, d_k, d_v)
        B_t = k_t.swapaxes(1, 2) * (w_t * v_t)  # (B, d_k, d_v)

        # ── State update and output ──────────────────────────────────────
        S_t = A_t @ S_t + B_t  # (B, d_k, d_v)   Eq. 10
        o_t = (q_t @ S_t).squeeze(1)  # (B, d_v)        o_t = S_t^T q_t

        outputs.append(o_t)

    return jnp.stack(outputs, axis=1)  # (B, L, d_v)


# ---------------------------------------------------------------------------
# Chunk-parallel formulation (naive O(C^3) per chunk, for learning only)
# ---------------------------------------------------------------------------


def chunked_forward(
    query: jax.Array,  # (B, L, d_k)
    key: jax.Array,  # (B, L, d_k)
    value: jax.Array,  # (B, L, d_v)
    beta: jax.Array,  # (B, L, d_k)  erase gate b_t
    gamma: jax.Array,  # (B, L, d_v)  write gate w_t  [C1]
    delta: jax.Array,  # (B, L, d_k)  per-channel decay alpha_t  [C1]
    chunk_size: int,
) -> jax.Array:  # (B, L, d_v)
    """
    Chunk-parallel implementation.

    For position r inside a chunk the state is expressed in closed form:

        S_r = (A_r A_{r-1} ... A_0) S_prev
            + sum_{i=0}^{r} (A_r ... A_{i+1}) B_i

    where S_prev is the state carried in from the previous chunk.
    All token states share the same S_prev, enabling parallel computation.

    Complexity: O((L/C) * C^3 * d_k^2)  -- very slow for large C.
    Exists only to make the math of the chunked recurrence transparent.
    """
    batch_size, seq_len, query_dim = query.shape
    value_dim = value.shape[-1]

    assert seq_len % chunk_size == 0, (  # [F3]
        f"seq_len ({seq_len}) must be divisible by chunk_size ({chunk_size})"
    )

    # Explicitly batched identity for matmuls inside the chunk loop.
    I = jnp.broadcast_to(
        jnp.eye(query_dim, dtype=query.dtype),
        (batch_size, query_dim, query_dim),
    )

    S_t = jnp.zeros((batch_size, query_dim, value_dim), dtype=query.dtype)
    outputs: list[jax.Array] = []
    num_chunks = seq_len // chunk_size

    for chunk_index in range(num_chunks):
        start = chunk_index * chunk_size

        # ── Slice the chunk, add a "1" axis at position 2 for matmul ────
        # Resulting shape: (B, C, 1, d_*)                            [F1]
        # (original comments said "(B, 1, d_*)" -- incorrect)
        q_c = jnp.expand_dims(
            query[:, start : start + chunk_size, :], axis=2
        )  # (B,C,1,d_k)
        k_c = jnp.expand_dims(
            key[:, start : start + chunk_size, :], axis=2
        )  # (B,C,1,d_k)
        v_c = jnp.expand_dims(
            value[:, start : start + chunk_size, :], axis=2
        )  # (B,C,1,d_v)
        b_c = jnp.expand_dims(
            beta[:, start : start + chunk_size, :], axis=2
        )  # (B,C,1,d_k)
        w_c = jnp.expand_dims(
            gamma[:, start : start + chunk_size, :], axis=2
        )  # (B,C,1,d_v)
        d_c = jnp.expand_dims(
            delta[:, start : start + chunk_size, :], axis=2
        )  # (B,C,1,d_k)

        # ── Build per-token A_r and B_r for r in [0, C) ─────────────────
        A_c: list[jax.Array] = []
        B_c: list[jax.Array] = []

        for r in range(chunk_size):
            # Index along the chunk axis (axis 1) to get single-token tensors.
            k_r = k_c[:, r, :, :]  # (B, 1, d_k)  [F5]
            v_r = v_c[:, r, :, :]  # (B, 1, d_v)
            b_r = b_c[:, r, :, :]  # (B, 1, d_k)
            w_r = w_c[:, r, :, :]  # (B, 1, d_v)
            d_r = d_c[:, r, :, :]  # (B, 1, d_k)

            A_r = (I - k_r.swapaxes(1, 2) * (b_r * k_r)) * d_r  # (B, d_k, d_k)
            B_r = k_r.swapaxes(1, 2) * (w_r * v_r)  # (B, d_k, d_v)

            A_c.append(A_r)
            B_c.append(B_r)

        # ── Compute S_r and o_r for each position in the chunk ───────────
        o_c: list[jax.Array] = []
        S_last = S_t  # will be overwritten; init avoids unbound-variable risk

        for r in range(chunk_size):
            # Prefix product P_r = A_r A_{r-1} ... A_0
            # Built left-to-right: A_c[i] @ prefix  =>  A_i(A_{i-1}(... I))
            prefix = I
            for i in range(r + 1):
                prefix = A_c[i] @ prefix  # (B, d_k, d_k)

            # Suffix-weighted write accumulation:
            #   accum = sum_{i=0}^{r} (A_r ... A_{i+1}) B_i
            # When i == r the suffix is I (no matrices to the right of B_r).
            accum = jnp.zeros(
                (batch_size, query_dim, value_dim), dtype=query.dtype
            )  # [F2]
            for i in range(r + 1):
                suffix = I
                for j in range(i + 1, r + 1):
                    suffix = A_c[j] @ suffix  # (B, d_k, d_k)
                accum = accum + suffix @ B_c[i]

            # Closed-form state at position r
            S_r = prefix @ S_t + accum  # (B, d_k, d_v)

            # Output token r
            o_r = (q_c[:, r, :, :] @ S_r).squeeze(1)  # (B, d_v)
            o_c.append(o_r)
            S_last = S_r  # track the last state in the chunk  [F4]

        S_t = S_last  # carry to the next chunk  [F4]
        outputs.extend(o_c)

    return jnp.stack(outputs, axis=1)  # (B, L, d_v)


# ---------------------------------------------------------------------------
# Core recurrence (mathematically correct in the original; kept intact)
# Parameter names are now aligned with paper notation:
#   b_t = erase gate  (paper: b_t ∈ [0,1]^{d_k})
#   w_t = write gate  (paper: w_t ∈ [0,1]^{d_v})
#   alpha_t = per-channel decay  (paper: α_t ∈ (0,1]^{d_k})
# ---------------------------------------------------------------------------


def chunked_forward_optimized(
    query: jax.Array,  # (B, L, d_k)
    key: jax.Array,  # (B, L, d_k)
    value: jax.Array,  # (B, L, d_v)
    b: jax.Array,  # (B, L, d_k) — erase gate
    w: jax.Array,  # (B, L, d_v) — write gate
    alpha: jax.Array,  # (B, L, d_k) — per-channel decay  ← was "delta"
    chunk_size: int,
) -> jax.Array:
    """
    Implements the Gated Delta Rule-2 recurrence (paper Eq. 10):

        S_t = (I − k_t (b_t ⊙ k_t)ᵀ) Diag(α_t) S_{t-1} + k_t (w_t ⊙ v_t)ᵀ
        o_t = Sₜᵀ q_t

    Uses a two-level loop:
      outer — jax.lax.scan across chunks (sequential across chunk boundaries)
      inner — jax.lax.associative_scan inside each chunk (parallel prefix)
    """
    batch_size, seq_len, dk = query.shape
    dv = value.shape[-1]
    num_chunks = seq_len // chunk_size
    assert seq_len % chunk_size == 0, (
        f"seq_len ({seq_len}) must be divisible by chunk_size ({chunk_size})"
    )

    # Reshape → (num_chunks, batch, chunk_size, dim) so scan iterates axis-0
    def to_scan(x):
        return x.reshape(batch_size, num_chunks, chunk_size, -1).swapaxes(0, 1)

    q_s = to_scan(query)
    k_s = to_scan(key)
    v_s = to_scan(value)
    b_s = to_scan(b)
    w_s = to_scan(w)
    a_s = to_scan(alpha)  # renamed from d_s

    I = jnp.eye(dk, dtype=query.dtype)

    # ── Associative scan combiner ──────────────────────────────────────────
    # Encodes the linear recurrence S_t = A_t S_{t-1} + B_t
    # (A1,B1) = prefix ending at t1; (A2,B2) = update from t1+1 to t2
    def combine(left, right):
        A1, B1 = left
        A2, B2 = right
        return A2 @ A1, A2 @ B1 + B2

    # ── Outer scan: one step = one chunk ──────────────────────────────────
    def chunk_step(S_prev, xs):
        q_c, k_c, v_c, b_c, w_c, a_c = xs

        # ── Build per-token A and B matrices (paper Eq. 10) ──────────────
        # erase factor: e_t = b_t ⊙ k_t
        # A_t = (I − k_t eₜᵀ) Diag(α_t)
        #     = (I − k_t (b_t⊙k_t)ᵀ) Diag(α_t)
        # [i,j] = (δ_{ij} − k_i·(b⊙k)_j) · α_j   ← column-wise scaling by α
        e_c = b_c * k_c  # (B, C, dk)
        outer_ke = jnp.einsum("bci,bcj->bcij", k_c, e_c)  # k eᵀ
        # Expand α to (B,C,1,dk) so it broadcasts column-wise
        A_c = (I - outer_ke) * jnp.expand_dims(a_c, axis=-2)  # (B,C,dk,dk)

        # write term: B_t = k_t (w_t ⊙ v_t)ᵀ = k_t zₜᵀ
        z_c = w_c * v_c  # (B, C, dv)
        B_c = jnp.einsum("bci,bcj->bcij", k_c, z_c)  # (B,C,dk,dv)

        # ── Associative scan over the chunk (parallel prefix) ─────────────
        A_ct = A_c.swapaxes(0, 1)  # (C, B, dk, dk)
        B_ct = B_c.swapaxes(0, 1)  # (C, B, dk, dv)
        A_cum_t, B_cum_t = jax.lax.associative_scan(combine, (A_ct, B_ct))
        A_cum = A_cum_t.swapaxes(0, 1)  # (B, C, dk, dk)
        B_cum = B_cum_t.swapaxes(0, 1)  # (B, C, dk, dv)

        # ── Compute all hidden states in this chunk ───────────────────────
        # S_r = A_cum_r S_prev + B_cum_r
        S_all = jnp.einsum("bcij,bjk->bcik", A_cum, S_prev) + B_cum
        # (B, C, dk, dv)

        # ── Outputs: o_r = Sᵀ_r q_r ─────────────────────────────────────
        o_c = jnp.einsum("bci,bcij->bcj", q_c, S_all)  # (B, C, dv)

        S_next = S_all[:, -1, :]  # last token's state
        return S_next, o_c

    S_init = jnp.zeros((batch_size, dk, dv), dtype=query.dtype)
    _, o_chunks = jax.lax.scan(
        chunk_step, S_init, (q_s, k_s, v_s, b_s, w_s, a_s)
    )  # o_chunks: (num_chunks, batch, chunk_size, dv)

    o = o_chunks.swapaxes(0, 1)  # (batch, num_chunks, chunk_size, dv)
    return o.reshape(batch_size, num_chunks * chunk_size, dv)


# ---------------------------------------------------------------------------
# Short causal convolution helper  [B3]
# Paper Fig.1: q, k, v each pass through a short causal conv + SiLU
# ---------------------------------------------------------------------------


class ShortCausalConv(nnx.Module):
    """
    Depthwise causal convolution with kernel size 4 (as used in GDN-2 / GDN).
    Equivalent to FLA's ShortConvolution in casual mode.
    """

    def __init__(self, dim: int, kernel_size: int = 4, rngs: nnx.Rngs = None):
        # Depthwise: groups = dim
        self.kernel_size = kernel_size
        self.dim = dim
        # Weight: (dim, 1, kernel_size)
        self.weight = nnx.Param(
            jax.random.normal(rngs.params(), (dim, kernel_size)) * 0.02
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: (B, L, dim)
        B, L, D = x.shape
        k = self.kernel_size
        # Causal padding: pad k-1 zeros at the start of the time axis
        x_pad = jnp.pad(x, ((0, 0), (k - 1, 0), (0, 0)))  # (B, L+k-1, D)
        # Depthwise conv via sliding window
        # weight: (D, k),  x_windows: (B, L, D, k)
        x_windows = jnp.stack(
            [x_pad[:, i : i + L, :] for i in range(k)], axis=-1
        )  # (B, L, D, k)
        out = jnp.einsum("bldk,dk->bld", x_windows, self.weight)
        return out


# ---------------------------------------------------------------------------
# Gated DeltaNet-2 token mixer  (GDN-2 block, Fig. 1)
# ---------------------------------------------------------------------------


class GatedDeltaNet2Layer(nnx.Module):
    """
    Full GDN-2 token mixer matching paper Figure 1 and Section 3.5:

      q, k paths: Linear → ShortConv → SiLU → L2-norm
      v   path:   Linear → ShortConv → SiLU
      α  (decay): dedicated projection → exp(-exp(a) * softplus(proj + bias))
      b  (erase): Linear → sigmoid
      w  (write): Linear → sigmoid
      output:     recurrent output → RMSNorm → SiLU-gate → Linear
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        chunk_size: int,
        conv_kernel: int = 4,
        rngs: nnx.Rngs = None,
    ):
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.chunk_size = chunk_size

        # ── Linear projections ──────────────────────────────────────────
        self.q_proj = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        self.k_proj = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)

        # ── Short causal convolutions  [B3] ────────────────────────────
        self.q_conv = ShortCausalConv(dim, conv_kernel, rngs=rngs)
        self.k_conv = ShortCausalConv(dim, conv_kernel, rngs=rngs)
        self.v_conv = ShortCausalConv(dim, conv_kernel, rngs=rngs)

        # ── Gate projections ────────────────────────────────────────────
        # b_t (erase): d_model → d_k per head, sigmoid → [0,1]^{d_k}
        self.b_proj = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        # w_t (write): d_model → d_v per head, sigmoid → [0,1]^{d_v}
        # [B5] renamed from gamma_proj
        self.w_proj = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)

        # ── Decay projection  [B2][B5] ──────────────────────────────────
        # Paper Eq. 12:  g_t = −exp(a) ⊙ softplus(W_f x_t + δ_bias)
        #                α_t = exp(g_t)  ∈ (0,1]^{d_k}
        # We store the learnable per-head-channel log-scale `a` separately.
        self.decay_proj = nnx.Linear(dim, dim, use_bias=True, rngs=rngs)
        # Learnable log-scale per key channel: shape (num_heads, head_dim)
        self.decay_log_scale = nnx.Param(jnp.zeros((num_heads, self.head_dim)))

        # ── Output gate + norm  [B4] ────────────────────────────────────
        # Paper Sec 3.5: "output is RMS-normalised, multiplied by a SiLU
        # output gate, and projected back to the model dimension"
        self.out_norm = nnx.RMSNorm(dim, rngs=rngs)
        self.out_gate_proj = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        self.o_proj = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)

    # ── helpers ──────────────────────────────────────────────────────────

    def _to_heads(self, x: jax.Array, batch: int, seq: int) -> jax.Array:
        """(B, L, dim) → (B*H, L, d_k)  for the flat recurrence call."""
        # (B, L, H, D_h) → (B, H, L, D_h) → (B*H, L, D_h)
        return (
            x.reshape(batch, seq, self.num_heads, self.head_dim)
            .swapaxes(1, 2)
            .reshape(batch * self.num_heads, seq, self.head_dim)
        )

    def _from_heads(self, x: jax.Array, batch: int, seq: int) -> jax.Array:
        """(B*H, L, d_h) → (B, L, dim)"""
        return (
            x.reshape(batch, self.num_heads, seq, self.head_dim)
            .swapaxes(1, 2)
            .reshape(batch, seq, self.num_heads * self.head_dim)
        )

    @staticmethod
    def _l2_norm(x: jax.Array, eps: float = 1e-6) -> jax.Array:
        return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), eps)

    # ── forward ──────────────────────────────────────────────────────────

    def __call__(self, x: jax.Array) -> jax.Array:
        B, L, _ = x.shape

        # ── q, k, v:  proj → conv → SiLU → L2 (q, k only)  [B1][B3] ──
        q = jax.nn.silu(self.q_conv(self.q_proj(x)))
        k = jax.nn.silu(self.k_conv(self.k_proj(x)))
        v = jax.nn.silu(self.v_conv(self.v_proj(x)))

        # L2-normalise both q AND k per head  [B1]
        q_h = q.reshape(B, L, self.num_heads, self.head_dim)
        k_h = k.reshape(B, L, self.num_heads, self.head_dim)
        q_h = self._l2_norm(q_h)
        k_h = self._l2_norm(k_h)

        # ── Erase gate b_t ∈ [0,1]^{d_k}  ─────────────────────────────
        b_h = jax.nn.sigmoid(self.b_proj(x)).reshape(
            B, L, self.num_heads, self.head_dim
        )

        # ── Write gate w_t ∈ [0,1]^{d_v}  [B5] ────────────────────────
        w_h = jax.nn.sigmoid(self.w_proj(x)).reshape(
            B, L, self.num_heads, self.head_dim
        )
        v_h = v.reshape(B, L, self.num_heads, self.head_dim)

        # ── Decay α_t ∈ (0,1]^{d_k}  [B2][B5] ──────────────────────────
        # Paper Eq.12:  g_t = −exp(a) ⊙ softplus(W_f x_t + bias)
        #               α_t = exp(g_t)
        raw = self.decay_proj(x)  # (B, L, dim)
        raw_h = raw.reshape(B, L, self.num_heads, self.head_dim)
        # decay_log_scale: (H, D_h), broadcast over B and L
        log_scale = jnp.exp(self.decay_log_scale)  # exp(a) ∈ (0,∞)
        # Compute g_t in higher precision for numerical stability (App D.1)
        g = -log_scale * jax.nn.softplus(raw_h.astype(jnp.float32))
        alpha_h = jnp.exp(g).astype(x.dtype)  # α_t ∈ (0,1]

        # ── Flatten (B,L,H,D_h) → (B*H, L, D_h) for the recurrence ────
        def flat(t):
            # t: (B, L, H, D_h)
            return t.swapaxes(1, 2).reshape(B * self.num_heads, L, self.head_dim)

        q_flat = flat(q_h)
        k_flat = flat(k_h)
        v_flat = flat(v_h)
        b_flat = flat(b_h)
        w_flat = flat(w_h)
        a_flat = flat(alpha_h)

        # ── Core recurrence ─────────────────────────────────────────────
        o_flat = chunked_forward_optimized(
            q_flat, k_flat, v_flat, b_flat, w_flat, a_flat, self.chunk_size
        )  # (B*H, L, D_h)

        # ── Recombine heads → (B, L, dim) ───────────────────────────────
        o = self._from_heads(o_flat, B, L)

        # ── Output: RMSNorm → SiLU-gate → projection  [B4] ─────────────
        # Paper Sec 3.5: output = (RMSNorm(o) * SiLU(gate)) W_o
        o_normed = self.out_norm(o)
        gate = jax.nn.silu(self.out_gate_proj(x))
        return self.o_proj(o_normed * gate)


# ---------------------------------------------------------------------------
# Transformer-style block (pre-norm, SwiGLU MLP)
# ---------------------------------------------------------------------------


class GatedDeltaNet2Block(nnx.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_dim: int,
        chunk_size: int,
        rngs: nnx.Rngs,
    ):
        self.norm1 = nnx.RMSNorm(dim, rngs=rngs)
        self.norm2 = nnx.RMSNorm(dim, rngs=rngs)
        self.mixer = GatedDeltaNet2Layer(dim, num_heads, chunk_size, rngs=rngs)
        # SwiGLU MLP
        self.mlp_gate = nnx.Linear(dim, mlp_dim, use_bias=False, rngs=rngs)
        self.mlp_up = nnx.Linear(dim, mlp_dim, use_bias=False, rngs=rngs)
        self.mlp_down = nnx.Linear(mlp_dim, dim, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x + self.mixer(self.norm1(x))
        h = self.norm2(x)
        x = x + self.mlp_down(jax.nn.silu(self.mlp_gate(h)) * self.mlp_up(h))
        return x


# ---------------------------------------------------------------------------
# Full language model
# ---------------------------------------------------------------------------


class GatedDeltaNet2(nnx.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        dim: int,
        num_heads: int,
        chunk_size: int,
        rngs: nnx.Rngs,
    ):
        self.embed = nnx.Embed(vocab_size, dim, rngs=rngs)
        self.blocks = nnx.Sequential(
            *[
                GatedDeltaNet2Block(
                    dim, num_heads, mlp_dim=dim * 4, chunk_size=chunk_size, rngs=rngs
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_f = nnx.RMSNorm(dim, rngs=rngs)
        self.lm_head = nnx.Linear(dim, vocab_size, use_bias=False, rngs=rngs)

    def __call__(self, input_ids: jax.Array) -> jax.Array:
        x = self.embed(input_ids)
        x = self.blocks(x)
        x = self.norm_f(x)
        return self.lm_head(x)
