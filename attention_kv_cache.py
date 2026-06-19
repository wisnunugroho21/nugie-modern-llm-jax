import jax
import jax.numpy as jnp
from flax import nnx


class GroupedQueryLatentAttentionV2(nnx.Module):
    def __init__(
        self,
        d_model: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dropout_rate: float,
        max_seq_length: int,
        rngs: nnx.Rngs,
        rope_base: float = 10000.0,
    ):
        if num_q_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})."
            )
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim ({head_dim}) must be even for RoPE.")

        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.group_size = num_q_heads // num_kv_heads
        self.max_seq_length = max_seq_length

        d_q = num_q_heads * head_dim
        d_kv = num_kv_heads * head_dim

        self.w_q_uk = nnx.Linear(d_model, d_q, use_bias=False, rngs=rngs)
        self.w_dkv = nnx.Linear(d_model, d_kv, use_bias=False, rngs=rngs)
        self.w_uv_o = nnx.Linear(d_q, d_model, use_bias=False, rngs=rngs)
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

        # --- RoPE PRECOMPUTATION ---
        half_dim = head_dim // 2
        inv_freq = 1.0 / (rope_base ** (jnp.arange(0, half_dim) / half_dim))
        positions = jnp.arange(self.max_seq_length)

        angles = jnp.outer(positions, inv_freq)
        angles = jnp.concatenate([angles, angles], axis=-1)

        # Shape: (1, 1, max_seq_length, head_dim)
        self.freqs_cos = jnp.cos(angles)[None, None, :, :]
        self.freqs_sin = jnp.sin(angles)[None, None, :, :]

        self.cache_k = None
        self.cache_v = None
        self.cache_index = None

    def init_cache(self, batch_size: int):
        self.cache_k = nnx.Cache(
            jnp.zeros(
                (batch_size, self.num_kv_heads, self.max_seq_length, self.head_dim)
            )
        )
        self.cache_v = nnx.Cache(
            jnp.zeros(
                (batch_size, self.num_kv_heads, self.max_seq_length, self.head_dim)
            )
        )

        # Track the current position for EACH sequence independently
        self.cache_index = nnx.Cache(jnp.zeros((batch_size,), dtype=jnp.int32))

    def apply_rope(self, x: jax.Array, position_ids: jax.Array) -> jax.Array:
        """
        Applies RoPE using explicit position IDs to account for padding shifts.
        position_ids shape: (batch_size, seq_len)
        """
        # Index directly into the precomputed frequencies
        # Resulting shape: (batch_size, seq_len, head_dim)
        cos = self.freqs_cos[0, 0, position_ids, :]
        sin = self.freqs_sin[0, 0, position_ids, :]

        # Add the num_heads dimension for broadcasting: (batch, 1, seq_len, head_dim)
        cos = cos[:, None, :, :]
        sin = sin[:, None, :, :]

        x1, x2 = jnp.split(x, 2, axis=-1)
        rotated_half_x = jnp.concatenate([-x2, x1], axis=-1)
        return (x * cos) + (rotated_half_x * sin)

    def __call__(
        self,
        x: jax.Array,
        attention_mask: jax.Array | None = None,
        position_ids: jax.Array | None = None,
        decode: bool = False,
        deterministic: bool = False,
    ) -> jax.Array:

        batch_size, seq_length, _ = x.shape

        if decode and (self.cache_k is None):
            raise RuntimeError(
                "You must call `init_cache(batch_size)` before decoding."
            )

        # 1. QUERIES & KV LATENTS
        q_heads = (
            self.w_q_uk(x)
            .reshape(batch_size, seq_length, self.num_q_heads, self.head_dim)
            .swapaxes(1, 2)
        )
        l_kv_heads = (
            self.w_dkv(x)
            .reshape(batch_size, seq_length, self.num_kv_heads, self.head_dim)
            .swapaxes(1, 2)
        )

        # 2. CACHE, RoPE, & POSITION TRACKING
        if decode:
            # start_pos is now a vector of shape (batch_size,)
            start_pos = self.cache_index.value

            if position_ids is None:
                # Expand to (batch_size, 1) for RoPE broadcasting
                position_ids = start_pos[:, None]

            q_heads = self.apply_rope(q_heads, position_ids)
            k_heads = self.apply_rope(l_kv_heads, position_ids)
            v_heads = l_kv_heads

            # Helper function to update a single sequence in the batch
            def update_single_cache(c, update, p):
                # c: (num_kv_heads, max_seq_len, head_dim)
                # update: (num_kv_heads, 1, head_dim)
                # p: scalar (current sequence length)
                return jax.lax.dynamic_update_slice(c, update, (0, p, 0))

            # Vectorize the helper across the batch dimension (axis 0 for all inputs)
            batched_update = jax.vmap(update_single_cache, in_axes=(0, 0, 0))

            self.cache_k.value = batched_update(self.cache_k.value, k_heads, start_pos)
            self.cache_v.value = batched_update(self.cache_v.value, v_heads, start_pos)

            k_full = self.cache_k.value
            v_full = self.cache_v.value

            # Increment all trackers by 1 simultaneously
            self.cache_index.value += 1

        else:
            if position_ids is None:
                position_ids = jnp.broadcast_to(
                    jnp.arange(seq_length), (batch_size, seq_length)
                )

            q_heads = self.apply_rope(q_heads, position_ids)
            k_heads = self.apply_rope(l_kv_heads, position_ids)
            v_heads = l_kv_heads

            if self.cache_k is not None:
                # For prefill, the update is uniform starting at index 0
                self.cache_k.value = jax.lax.dynamic_update_slice(
                    self.cache_k.value, k_heads, (0, 0, 0, 0)
                )
                self.cache_v.value = jax.lax.dynamic_update_slice(
                    self.cache_v.value, v_heads, (0, 0, 0, 0)
                )

                # Determine the true length of each sequence based on the padding mask
                if attention_mask is not None:
                    # Sum the valid tokens (1s) to find exactly where generation should begin
                    self.cache_index.value = jnp.sum(attention_mask, axis=-1).astype(
                        jnp.int32
                    )
                else:
                    self.cache_index.value = jnp.full(
                        (batch_size,), seq_length, dtype=jnp.int32
                    )

            k_full = k_heads
            v_full = v_heads

        # 3. EXPLICITLY REPEAT KV HEADS
        k_repeated = k_full.repeat(self.group_size, axis=1)
        v_repeated = v_full.repeat(self.group_size, axis=1)

        # 4. ATTENTION SCORES
        qk_t = q_heads @ k_repeated.swapaxes(-1, -2)
        scaled_logits = qk_t / jnp.sqrt(self.head_dim)

        # 5. DYNAMIC MASKING (Causal + Padding)
        if decode:
            # start_pos: (batch_size,) -> (batch_size, 1) to compare against arange
            # Resulting mask: (batch_size, max_seq_length)
            mask = jnp.arange(self.max_seq_length)[None, :] <= start_pos[:, None]

            # Expand to (batch_size, 1, 1, max_seq_length) to match scaled_logits
            mask = mask[:, None, None, :]
        else:
            # Base causal mask for prefill
            mask = jnp.tril(jnp.ones((seq_length, seq_length), dtype=bool))[
                None, None, :, :
            ]

        # Combine with explicit user padding mask (0 for pad, 1 for real)
        if attention_mask is not None:
            # attention_mask shape expected: (batch_size, kv_len)
            # Expand to (batch_size, 1, 1, kv_len) to broadcast across heads and queries
            mask = mask & attention_mask[:, None, None, :]

        # Apply final combined mask
        scaled_logits = jnp.where(mask, scaled_logits, -jnp.inf)

        # 6. SOFTMAX & OUTPUT
        a = jax.nn.softmax(scaled_logits, axis=-1)
        dropped_a = self.dropout(a, deterministic=deterministic)

        weighted_heads = dropped_a @ v_repeated
        weighted_reshaped = weighted_heads.swapaxes(1, 2)

        weighted_latents = weighted_reshaped.reshape(
            batch_size, seq_length, self.num_q_heads * self.head_dim
        )

        return self.w_uv_o(weighted_latents)
