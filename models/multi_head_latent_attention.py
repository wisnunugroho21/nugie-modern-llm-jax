import flax.nnx as nnx
import jax
import jax.numpy as jnp


class MultiHeadLatentAttentionV1(nnx.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kv_compress_dim: int,
        query_compress_dim: int,
        dropout_rate: float,
        rngs: nnx.Rngs,
    ) -> None:
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.kv_compress_dim = kv_compress_dim
        self.query_compress_dim = query_compress_dim
        self.head_dim = embed_dim // num_heads

        self.kv_compress_nn = nnx.Linear(
            self.embed_dim, self.kv_compress_dim, rngs=rngs
        )
        self.query_compress_nn = nnx.Linear(
            self.embed_dim, self.query_compress_dim, rngs=rngs
        )

        self.key_decompress_nn = nnx.Linear(
            self.kv_compress_dim, self.num_heads * self.head_dim, rngs=rngs
        )
        self.value_decompress_nn = nnx.Linear(
            self.kv_compress_dim, self.num_heads * self.head_dim, rngs=rngs
        )
        self.query_decompress_nn = nnx.Linear(
            self.query_compress_dim, self.num_heads * self.head_dim, rngs=rngs
        )

        self.out_nn = nnx.Linear(
            self.num_heads * self.head_dim, self.embed_dim, rngs=rngs
        )

        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        batch_size, seq_length, _ = x.shape

        kv_compress = self.kv_compress_nn(x)
        query_compress = self.query_compress_nn(x)

        query = self.query_decompress_nn(query_compress)
        key = self.key_decompress_nn(kv_compress)
        value = self.value_decompress_nn(kv_compress)

        query = query.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        key = key.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        value = value.reshape(batch_size, seq_length, self.num_heads, self.head_dim)

        query = query.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)

        causal_mask = jnp.tril(jnp.ones((seq_length, seq_length), dtype=bool))

        scores = query @ key.swapaxes(-2, -1) / jnp.sqrt(self.head_dim)
        scores = jnp.where(
            causal_mask[None, None, :seq_length, :seq_length], scores, -jnp.inf
        )

        weights = jax.nn.softmax(scores, axis=-1)
        weights = self.dropout(weights)

        context = weights @ value
        context = context.transpose(0, 2, 1, 3)
        context = context.reshape(
            batch_size, seq_length, self.num_heads * self.head_dim
        )

        return self.out_nn(context)


class MultiHeadLatentAttentionV2(nnx.Module):
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
    d_model     : embedding dimension  (diagram: 7168)
    d_latent    : latent dimension     (diagram:  576)
    rngs        : Flax NNX random number generators
    """

    def __init__(self, d_model: int, d_latent: int, rngs: nnx.Rngs):
        # Corresponds to W_Q W_UK^T (e.g., 7168 x 576)
        self.w_q_uk = nnx.Linear(d_model, d_latent, use_bias=False, rngs=rngs)

        # Corresponds to W_DKV (e.g., 7168 x 576)
        self.w_dkv = nnx.Linear(d_model, d_latent, use_bias=False, rngs=rngs)

        # Corresponds to W_UV W_O (e.g., 576 x 7168)
        self.w_uv_o = nnx.Linear(d_latent, d_model, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array, mask: jax.Array | None = None) -> jax.Array:
        """
        Forward pass.

        Returns
        -------
        out   : final output — same shape as X
        """

        # x shape: (batch_size, seq_len, d_model)
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


class MultiHeadLatentAttentionV3(nnx.Module):
    def __init__(self, d_model: int, d_latent: int, rngs: nnx.Rngs):
        # Corresponds to W_Q W_UK^T (e.g., 7168 x 576)
        self.w_q_uk = nnx.Linear(d_model, d_latent, use_bias=False, rngs=rngs)

        # Corresponds to W_DKV (e.g., 7168 x 576)
        self.w_dkv = nnx.Linear(d_model, d_latent, use_bias=False, rngs=rngs)

        # Corresponds to W_UV W_O (e.g., 576 x 7168)
        self.w_uv_o = nnx.Linear(d_latent, d_model, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array, mask: jax.Array | None = None) -> jax.Array:
        # x shape: (batch_size, seq_len, d_model)
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


class MultiHeadLatentAttentionV4(nnx.Module):
    def __init__(self, d_model: int, d_latent: int, num_heads: int, rngs: nnx.Rngs):
        # Ensure the latent dimension is perfectly divisible by the number of heads
        if d_latent % num_heads != 0:
            raise ValueError(
                f"d_latent ({d_latent}) must be divisible by num_heads ({num_heads})."
            )

        self.d_latent = d_latent
        self.num_heads = num_heads
        self.head_dim = d_latent // num_heads

        # Corresponds to W_Q W_UK^T
        self.w_q_uk = nnx.Linear(d_model, d_latent, use_bias=False, rngs=rngs)

        # Corresponds to W_DKV
        self.w_dkv = nnx.Linear(d_model, d_latent, use_bias=False, rngs=rngs)

        # Corresponds to W_UV W_O
        self.w_uv_o = nnx.Linear(d_latent, d_model, use_bias=False, rngs=rngs)

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
        output = self.w_uv_o(weighted_latents)  # Shape: (batch, seq_len, d_model)

        return output
