import jax
import jax.numpy as jnp
from flax import nnx


class GroupedQueryLatentAttentionV1(nnx.Module):
    def __init__(
        self,
        embed_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rngs: nnx.Rngs,
    ):
        if num_q_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})."
            )

        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.group_size = num_q_heads // num_kv_heads

        # Projections
        d_q = num_q_heads * head_dim
        d_kv = num_kv_heads * head_dim

        self.w_q_uk = nnx.Linear(embed_dim, d_q, use_bias=False, rngs=rngs)
        self.w_dkv = nnx.Linear(embed_dim, d_kv, use_bias=False, rngs=rngs)
        self.w_uv_o = nnx.Linear(d_q, embed_dim, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array, mask: jax.Array | None = None) -> jax.Array:
        batch_size, seq_len, _ = x.shape

        # 1. QUERIES (N Heads)
        q_latent = self.w_q_uk(x)

        # Shape: (batch, num_q_heads, seq_len, head_dim)
        q_reshaped = q_latent.reshape(
            batch_size, seq_len, self.num_q_heads, self.head_dim
        )

        # Transpose to (batch, num_q_heads, seq_len, head_dim)
        q_heads = q_reshaped.swapaxes(1, 2)

        # 2. KV LATENTS (M Heads)
        l_kv = self.w_dkv(x)

        # Shape: (batch, num_kv_heads, seq_len, head_dim)
        l_kv_reshaped = l_kv.reshape(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        )

        # Transpose to (batch, num_kv_heads, seq_len, head_dim)
        l_kv_heads = l_kv_reshaped.swapaxes(1, 2)

        # 3. EXPLICITLY REPEAT KV HEADS
        # Repeat the M heads along the num_heads axis (axis=1) by the group_size
        # The new shape becomes identically: (batch, num_q_heads, seq_len, head_dim)
        l_kv_repeated = l_kv_heads.repeat(self.group_size, axis=1)

        # 4. ATTENTION SCORES (Standard MHA calculation now!)
        # Q: (batch, num_q_heads, seq_len, head_dim)
        # K transposed: (batch, num_q_heads, head_dim, seq_len)
        qk_t = q_heads @ l_kv_repeated.swapaxes(-1, -2)

        # 5. MASK
        scaled_logits = qk_t / jnp.sqrt(self.head_dim)
        if mask is not None:
            scaled_logits = scaled_logits + mask

        # 6. SOFTMAX
        a = jax.nn.softmax(scaled_logits, axis=-1)

        # 7. WEIGHTED LATENTS
        # Both tensors now natively align on the num_q_heads dimension
        # A: (batch, num_q_heads, seq_len, seq_len)
        # V: (batch, num_q_heads, seq_len, head_dim)
        weighted_heads = a @ l_kv_repeated

        # 8. RECOMBINE HEADS
        # Swap back to (batch, seq_len, num_q_heads, head_dim)
        weighted_reshaped = weighted_heads.swapaxes(1, 2)

        # Flatten: (batch, seq_len, num_q_heads * head_dim)
        weighted_latents = weighted_reshaped.reshape(
            batch_size, seq_len, self.num_q_heads * self.head_dim
        )

        # 9. OUTPUT PROJECTION
        output = self.w_uv_o(weighted_latents)

        return output


class GroupedQueryLatentAttentionV2(nnx.Module):
    def __init__(
        self,
        embed_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dropout_rate: float,
        seq_length: int,
        rngs: nnx.Rngs,
    ):
        if num_q_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})."
            )

        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.group_size = num_q_heads // num_kv_heads

        # Projections
        d_q = num_q_heads * head_dim
        d_kv = num_kv_heads * head_dim

        self.w_q_uk = nnx.Linear(embed_dim, d_q, use_bias=False, rngs=rngs)
        self.w_dkv = nnx.Linear(embed_dim, d_kv, use_bias=False, rngs=rngs)
        self.w_uv_o = nnx.Linear(d_q, embed_dim, use_bias=False, rngs=rngs)

        self.causal_mask = jnp.tril(jnp.ones((seq_length, seq_length), dtype=bool))
        self.dropout = nnx.Dropout(
            dropout_rate, rngs=rngs
        )  # Optional dropout layer for attention weights

    def __call__(self, x: jax.Array, deterministic: bool = False) -> jax.Array:
        batch_size, seq_length, _ = x.shape

        # 1. QUERIES (N Heads)
        q_latent = self.w_q_uk(x)

        # Shape: (batch, num_q_heads, seq_len, head_dim)
        q_reshaped = q_latent.reshape(
            batch_size, seq_length, self.num_q_heads, self.head_dim
        )

        # Transpose to (batch, num_q_heads, seq_len, head_dim)
        q_heads = q_reshaped.swapaxes(1, 2)

        # 2. KV LATENTS (M Heads)
        l_kv = self.w_dkv(x)

        # Shape: (batch, num_kv_heads, seq_len, head_dim)
        l_kv_reshaped = l_kv.reshape(
            batch_size, seq_length, self.num_kv_heads, self.head_dim
        )

        # Transpose to (batch, num_kv_heads, seq_len, head_dim)
        l_kv_heads = l_kv_reshaped.swapaxes(1, 2)

        # 3. EXPLICITLY REPEAT KV HEADS
        # Repeat the M heads along the num_heads axis (axis=1) by the group_size
        # The new shape becomes identically: (batch, num_q_heads, seq_len, head_dim)
        l_kv_repeated = l_kv_heads.repeat(self.group_size, axis=1)

        # 4. ATTENTION SCORES (Standard MHA calculation now!)
        # Q: (batch, num_q_heads, seq_len, head_dim)
        # K transposed: (batch, num_q_heads, head_dim, seq_len)
        qk_t = q_heads @ l_kv_repeated.swapaxes(-1, -2)

        # 5. MASK
        scaled_logits = qk_t / jnp.sqrt(self.head_dim)
        scaled_logits = jnp.where(
            self.causal_mask[None, None, :seq_length, :seq_length],
            scaled_logits,
            -jnp.inf,
        )

        # 6. SOFTMAX
        a = jax.nn.softmax(scaled_logits, axis=-1)

        # Apply dropout to attention weights if not deterministic
        dropped_a = self.dropout(a, deterministic=deterministic)

        # 7. WEIGHTED LATENTS
        # Both tensors now natively align on the num_q_heads dimension
        # A: (batch, num_q_heads, seq_len, seq_len)
        # V: (batch, num_q_heads, seq_len, head_dim)
        weighted_heads = dropped_a @ l_kv_repeated

        # 8. RECOMBINE HEADS
        # Swap back to (batch, seq_len, num_q_heads, head_dim)
        weighted_reshaped = weighted_heads.swapaxes(1, 2)

        # Flatten: (batch, seq_len, num_q_heads * head_dim)
        weighted_latents = weighted_reshaped.reshape(
            batch_size, seq_length, self.num_q_heads * self.head_dim
        )

        # 9. OUTPUT PROJECTION
        output = self.w_uv_o(weighted_latents)

        return output
    
class GroupedQueryLatentAttention(nnx.Module):
    def __init__(
        self,
        embed_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dropout_rate: float,
        seq_length: int,
        rngs: nnx.Rngs,
    ):
        if num_q_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})."
            )

        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.group_size = num_q_heads // num_kv_heads

        # Projections
        d_q = num_q_heads * head_dim
        d_kv = num_kv_heads * head_dim

        self.w_q_uk = nnx.Linear(embed_dim, d_q, use_bias=False, rngs=rngs)
        self.w_dkv = nnx.Linear(embed_dim, d_kv, use_bias=False, rngs=rngs)
        self.w_uv_o = nnx.Linear(d_q, embed_dim, use_bias=False, rngs=rngs)

        self.causal_mask = jnp.tril(jnp.ones((seq_length, seq_length), dtype=bool))
        self.dropout = nnx.Dropout(
            dropout_rate, rngs=rngs
        )  # Optional dropout layer for attention weights

    def __call__(self, x: jax.Array, deterministic: bool = False) -> jax.Array:
        batch_size, seq_length, _ = x.shape

        # 1. QUERIES (N Heads)
        q_latent = self.w_q_uk(x)

        # Shape: (batch, num_q_heads, seq_len, head_dim)
        q_reshaped = q_latent.reshape(
            batch_size, seq_length, self.num_q_heads, self.head_dim
        )

        # Transpose to (batch, num_q_heads, seq_len, head_dim)
        q_heads = q_reshaped.swapaxes(1, 2)

        # 2. KV LATENTS (M Heads)
        l_kv = self.w_dkv(x)

        # Shape: (batch, num_kv_heads, seq_len, head_dim)
        l_kv_reshaped = l_kv.reshape(
            batch_size, seq_length, self.num_kv_heads, self.head_dim
        )

        # Transpose to (batch, num_kv_heads, seq_len, head_dim)
        l_kv_heads = l_kv_reshaped.swapaxes(1, 2)

        # 3. EXPLICITLY REPEAT KV HEADS
        # Repeat the M heads along the num_heads axis (axis=1) by the group_size
        # The new shape becomes identically: (batch, num_q_heads, seq_len, head_dim)
        l_kv_repeated = l_kv_heads.repeat(self.group_size, axis=1)

        # 4. ATTENTION SCORES (Standard MHA calculation now!)
        # Q: (batch, num_q_heads, seq_len, head_dim)
        # K transposed: (batch, num_q_heads, head_dim, seq_len)
        qk_t = jnp.einsum("bhqd, bhkd -> bhqk", q_heads, l_kv_repeated)

        # 5. MASK
        scaled_logits = qk_t / jnp.sqrt(self.head_dim)
        scaled_logits = jnp.where(
            self.causal_mask[None, None, :seq_length, :seq_length],
            scaled_logits,
            -jnp.inf,
        )

        # 6. SOFTMAX
        a = jax.nn.softmax(scaled_logits, axis=-1)

        # Apply dropout to attention weights if not deterministic
        dropped_a = self.dropout(a, deterministic=deterministic)

        # 7. WEIGHTED LATENTS
        # Both tensors now natively align on the num_q_heads dimension
        # A: (batch, num_q_heads, seq_len, seq_len)
        # V: (batch, num_q_heads, seq_len, head_dim)
        weighted_heads = jnp.einsum("bhqk, bhvd -> bhqd", dropped_a, l_kv_repeated)

        # 8. RECOMBINE HEADS
        # Swap back to (batch, seq_len, num_q_heads, head_dim)
        weighted_reshaped = weighted_heads.swapaxes(1, 2)

        # Flatten: (batch, seq_len, num_q_heads * head_dim)
        weighted_latents = weighted_reshaped.reshape(
            batch_size, seq_length, self.num_q_heads * self.head_dim
        )

        # 9. OUTPUT PROJECTION
        output = self.w_uv_o(weighted_latents)

        return output