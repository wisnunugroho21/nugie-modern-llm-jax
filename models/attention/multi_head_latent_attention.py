import flax.nnx as nnx
import jax
import jax.numpy as jnp


class MultiHeadLatentAttention(nnx.Module):
    def __init__(self, embed_dim: int, d_latent: int, num_heads: int, rngs: nnx.Rngs):
        # Ensure the latent dimension is perfectly divisible by the number of heads
        if d_latent % num_heads != 0:
            raise ValueError(
                f"d_latent ({d_latent}) must be divisible by num_heads ({num_heads})."
            )

        self.d_latent = d_latent
        self.num_heads = num_heads
        self.head_dim = d_latent // num_heads

        # Corresponds to W_Q W_UK^T
        self.w_q_uk = nnx.Linear(embed_dim, d_latent, use_bias=False, rngs=rngs)

        # Corresponds to W_DKV
        self.w_dkv = nnx.Linear(embed_dim, d_latent, use_bias=False, rngs=rngs)

        # Corresponds to W_UV W_O
        self.w_uv_o = nnx.Linear(d_latent, embed_dim, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array, mask: jax.Array | None = None) -> jax.Array:
        batch_size, seq_len, _ = x.shape

        # 1. QUERIES PROJECTED TO LATENT SPACE
        q_latent = self.w_q_uk(x)  # Shape: (batch, seq_len, d_latent)

        # 2. KEY & VALUE LATENTS
        l_kv = self.w_dkv(x)  # Shape: (batch, seq_len, d_latent)

        # 3. SPLIT INTO HEADS
        # Reshape to (batch, seq_len, num_heads, head_dim)
        q_reshaped = q_latent.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        l_kv_reshaped = l_kv.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to isolate heads in the batch dimensions: (batch, num_heads, seq_len, head_dim)
        q_heads = jnp.swapaxes(q_reshaped, 1, 2)
        l_kv_heads = jnp.swapaxes(l_kv_reshaped, 1, 2)

        # 4. ATTENTION SCORES: QK^T per head
        # Transpose the last two dimensions of l_kv_heads for matrix multiplication
        qk_t = q_heads @ jnp.swapaxes(
            l_kv_heads, -1, -2
        )  # Shape: (batch, num_heads, seq_len, seq_len)

        # 5. MASK & SOFTMAX
        # Scale by sqrt(head_dim) instead of the full latent dim
        scaled_logits = qk_t / jnp.sqrt(self.head_dim)

        if mask is not None:
            # Mask (seq_len, seq_len) automatically broadcasts across batch and num_heads
            scaled_logits = scaled_logits + mask

        a = jax.nn.softmax(scaled_logits, axis=-1)

        # 6. WEIGHTED LATENTS per head
        # (batch, num_heads, seq_len, seq_len) @ (batch, num_heads, seq_len, head_dim)
        weighted_latents_heads = (
            a @ l_kv_heads
        )  # Shape: (batch, num_heads, seq_len, head_dim)

        # 7. RECOMBINE HEADS
        # Transpose back to (batch, seq_len, num_heads, head_dim)
        weighted_latents_reshaped = jnp.swapaxes(weighted_latents_heads, 1, 2)

        # Flatten the last two dimensions back to (batch, seq_len, d_latent)
        weighted_latents = weighted_latents_reshaped.reshape(
            batch_size, seq_len, self.d_latent
        )

        # 8. OUTPUT PROJECTION
        output = self.w_uv_o(weighted_latents)  # Shape: (batch, seq_len, embed_dim)

        return output
