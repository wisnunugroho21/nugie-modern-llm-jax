import flax.nnx as nnx
import jax
import jax.numpy as jnp


class MultiHeadLatentAttentionV1(nnx.Module):
    """
    Multi-Head Latent Attention (MLA) in Flax NNX.

    Five equations from the diagram:

      (1)  L_KV  = X W_DKV                      [KV down-projection]
      (2)  Q     = X (W_Q W_UK^T)               [Query latent]
      (3)  QK^T  = Q L_KV^T / sqrt(d_latent)    [Scaled scores]
      (4)  A     = Softmax( mask(QK^T) )        [Attention weights]
      (5)  out   = (A L_KV)(W_UV W_O)           [Weighted output]

    Parameters
    ----------
    embed_dim     : embedding dimension  (diagram: 7168)
    d_latent    : latent dimension     (diagram:  576)
    rngs        : Flax NNX random number generators
    """

    def __init__(self, embed_dim: int, d_latent: int, rngs: nnx.Rngs):
        # Corresponds to W_Q W_UK^T (e.g., 7168 x 576)
        self.w_q_uk = nnx.Linear(embed_dim, d_latent, use_bias=False, rngs=rngs)

        # Corresponds to W_DKV (e.g., 7168 x 576)
        self.w_dkv = nnx.Linear(embed_dim, d_latent, use_bias=False, rngs=rngs)

        # Corresponds to W_UV W_O (e.g., 576 x 7168)
        self.w_uv_o = nnx.Linear(d_latent, embed_dim, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array, mask: jax.Array | None = None) -> jax.Array:
        """
        Forward pass.

        Returns
        -------
        out   : final output — same shape as X
        """

        # x shape: (batch_size, seq_len, embed_dim)
        # d is the latent dimension for scaling
        d = self.w_q_uk.out_features

        # 1. QUERIES PROJECTED TO LATENT SPACE
        # X(W_Q W_UK^T)
        q_latent = self.w_q_uk(x)

        # 2. KEY & VALUE LATENTS (Shared across all heads & cached)
        # L_KV = X W_DKV
        l_kv = self.w_dkv(x)

        # 3. ATTENTION SCORES: QK^T
        # Using einsum to handle optional batch dimensions cleanly (...id, ...jd -> ...ij)
        qk_t = q_latent @ l_kv.swapaxes(-2, -1)

        # 4. MASK & SOFTMAX
        # A = Softmax(QK^T / sqrt(d))
        scaled_logits = qk_t / jnp.sqrt(d)

        if mask is not None:
            # Assumes an additive mask where masked positions are large negative numbers (e.g. -1e9)
            scaled_logits = scaled_logits + mask

        a = jax.nn.softmax(scaled_logits, axis=-1)

        # 5. WEIGHTED LATENTS
        # A L_KV
        weighted_latents = a @ l_kv

        # 6. OUTPUT PROJECTION
        # (A L_KV)(W_UV W_O)
        output = self.w_uv_o(weighted_latents)

        return output


class MultiHeadLatentAttentionV2(nnx.Module):
    def __init__(self, embed_dim: int, d_latent: int, rngs: nnx.Rngs):
        # Corresponds to W_Q W_UK^T (e.g., 7168 x 576)
        self.w_q_uk = nnx.Linear(embed_dim, d_latent, use_bias=False, rngs=rngs)

        # Corresponds to W_DKV (e.g., 7168 x 576)
        self.w_dkv = nnx.Linear(embed_dim, d_latent, use_bias=False, rngs=rngs)

        # Corresponds to W_UV W_O (e.g., 576 x 7168)
        self.w_uv_o = nnx.Linear(d_latent, embed_dim, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array, mask: jax.Array | None = None) -> jax.Array:
        # x shape: (batch_size, seq_len, embed_dim)
        # d is the latent dimension for scaling
        d = self.w_q_uk.out_features

        # 1. QUERIES PROJECTED TO LATENT SPACE
        # X(W_Q W_UK^T)
        q_latent = self.w_q_uk(x)

        # 2. KEY & VALUE LATENTS (Shared across all heads & cached)
        # L_KV = X W_DKV
        l_kv = self.w_dkv(x)

        # 3. ATTENTION SCORES: QK^T
        # Using einsum to handle optional batch dimensions cleanly (...id, ...jd -> ...ij)
        qk_t = jnp.einsum("...id,...jd->...ij", q_latent, l_kv)

        # 4. MASK & SOFTMAX
        # A = Softmax(QK^T / sqrt(d))
        scaled_logits = qk_t / jnp.sqrt(d)

        if mask is not None:
            # Assumes an additive mask where masked positions are large negative numbers (e.g. -1e9)
            scaled_logits = scaled_logits + mask

        a = jax.nn.softmax(scaled_logits, axis=-1)

        # 5. WEIGHTED LATENTS
        # A L_KV
        weighted_latents = jnp.einsum("...ij,...jd->...id", a, l_kv)

        # 6. OUTPUT PROJECTION
        # (A L_KV)(W_UV W_O)
        output = self.w_uv_o(weighted_latents)

        return output
