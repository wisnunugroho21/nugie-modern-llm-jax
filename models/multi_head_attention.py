import flax.nnx as nnx
import jax
import jax.numpy as jnp

class MultiHeadAttentionV1(nnx.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        dropout_rate: float,
        rngs: nnx.Rngs,
    ) -> None:
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.query_nn = nnx.Linear(
            self.embed_dim, self.num_heads * self.head_dim, rngs=rngs
        )
        self.key_nn = nnx.Linear(
            self.embed_dim, self.num_heads * self.head_dim, rngs=rngs
        )
        self.value_nn = nnx.Linear(
            self.embed_dim, self.num_heads * self.head_dim, rngs=rngs
        )
        self.out_nn = nnx.Linear(
            self.num_heads * self.head_dim, self.embed_dim, rngs=rngs
        )
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        batch_size, seq_length, _ = x.shape

        query = self.query_nn(x)
        key = self.key_nn(x)
        value = self.value_nn(x)

        query = query.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        key = key.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        value = value.reshape(batch_size, seq_length, self.num_heads, self.head_dim)

        query = query.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)

        causal_mask = jnp.tril(jnp.ones((seq_length, seq_length), dtype=bool))

        scores = query @ key.swapaxes(-2, -1) / jnp.sqrt(self.head_dim)
        scores = jnp.where(causal_mask[None, None, :seq_length, :seq_length], scores, -jnp.inf)

        weights = jax.nn.softmax(scores, axis=-1)
        weights = self.dropout(weights)

        context = weights @ value
        context = context.transpose(0, 2, 1, 3)
        context = context.reshape(
            batch_size, seq_length, self.num_heads * self.head_dim
        )

        return self.out_nn(context)
    
class MultiHeadAttentionV2(nnx.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        dropout_rate: float,
        seq_length: int,
        rngs: nnx.Rngs,
    ) -> None:
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.qkv_nn = nnx.Linear(
            self.embed_dim, 3 * self.num_heads * self.head_dim, rngs=rngs
        )

        self.out_nn = nnx.Linear(
            self.num_heads * self.head_dim, self.embed_dim, rngs=rngs
        )

        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

        self.causal_mask = jnp.tril(jnp.ones((seq_length, seq_length), dtype=bool))
        self.scale = 1.0 / jnp.sqrt(self.head_dim)

    def __call__(self, x: jax.Array) -> jax.Array:
        batch_size, seq_length, _ = x.shape

        qkv = self.qkv_nn(x)
        query, key, value = jnp.split(qkv, 3, axis=-1)

        query = query.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        key = key.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        value = value.reshape(batch_size, seq_length, self.num_heads, self.head_dim)

        query = query.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)

        scores = query @ key.swapaxes(-2, -1) * self.scale
        scores = jnp.where(self.causal_mask[None, None, :seq_length, :seq_length], scores, -jnp.inf)

        weights = jax.nn.softmax(scores, axis=-1)
        weights = self.dropout(weights)

        context = weights @ value
        context = context.transpose(0, 2, 1, 3)
        context = context.reshape(
            batch_size, seq_length, self.num_heads * self.head_dim
        )

        return self.out_nn(context)
    
class MultiHeadAttentionV3(nnx.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        dropout_rate: float,
        seq_length: int,
        rngs: nnx.Rngs,
    ) -> None:
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.qkv_nn = nnx.Linear(
            self.embed_dim, 3 * self.num_heads * self.head_dim, rngs=rngs
        )

        self.out_nn = nnx.Linear(
            self.num_heads * self.head_dim, self.embed_dim, rngs=rngs
        )

        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

        self.causal_mask = jnp.tril(jnp.ones((seq_length, seq_length), dtype=bool))
        self.scale = 1.0 / jnp.sqrt(self.head_dim)

    def __call__(self, x: jax.Array) -> jax.Array:
        batch_size, seq_length, _ = x.shape

        qkv = self.qkv_nn(x)
        query, key, value = jnp.split(qkv, 3, axis=-1)

        query = query.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        key = key.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        value = value.reshape(batch_size, seq_length, self.num_heads, self.head_dim)

        query = query.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)

        scores = jnp.einsum('bhqd, bhkd -> bhqk', query, key) * self.scale
        scores = jnp.where(self.causal_mask[None, None, :seq_length, :seq_length], scores, -jnp.inf)

        weights = jax.nn.softmax(scores, axis=-1)
        weights = self.dropout(weights)

        context = jnp.einsum('bhqk, bhvd -> bhqd', weights, value)
        context = context.transpose(0, 2, 1, 3)
        context = context.reshape(
            batch_size, seq_length, self.num_heads * self.head_dim
        )

        return self.out_nn(context)