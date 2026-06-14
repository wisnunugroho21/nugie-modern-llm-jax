import flax.nnx as nnx
import jax
import jax.numpy as jnp

import tiktoken
from transformers import AutoTokenizer, PreTrainedTokenizer


class GroupQueryAttention(nnx.Module):
    def __init__(
        self,
        embed_dim: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dropout_rate: float,
        seq_length: int,
        rngs: nnx.Rngs,
    ) -> None:
        self.embed_dim = embed_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.query_nn = nnx.Linear(
            self.embed_dim, self.num_query_heads * self.head_dim, rngs=rngs
        )
        self.key_nn = nnx.Linear(
            self.embed_dim, self.num_kv_heads * self.head_dim, rngs=rngs
        )
        self.value_nn = nnx.Linear(
            self.embed_dim, self.num_kv_heads * self.head_dim, rngs=rngs
        )
        self.out_nn = nnx.Linear(
            self.num_query_heads * self.head_dim, self.embed_dim, rngs=rngs
        )
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

        self.causal_mask = jnp.tril(jnp.ones((seq_length, seq_length), dtype=bool))
        self.scale = 1.0 / jnp.sqrt(self.head_dim)
        

    def __call__(self, x: jax.Array) -> jax.Array:
        batch_size, seq_length, _ = x.shape

        if self.num_query_heads % self.num_kv_heads != 0:
            raise ValueError("num_query_heads must be divisible by num_kv_heads")

        query = self.query_nn(x)
        key = self.key_nn(x)
        value = self.value_nn(x)

        query = query.reshape(
            batch_size, seq_length, self.num_query_heads, self.head_dim
        )
        key = key.reshape(batch_size, seq_length, self.num_kv_heads, self.head_dim)
        value = value.reshape(batch_size, seq_length, self.num_kv_heads, self.head_dim)

        kv_repeat = self.num_query_heads // self.num_kv_heads
        key = key.repeat(kv_repeat, axis=-2)
        value = value.repeat(kv_repeat, axis=-2)

        query = query.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)

        scores = query @ key.swapaxes(-1, -2)
        scores = jnp.where(self.causal_mask[None, None, :, :], scores, -1e30)
        scores = scores * self.scale

        weights = jax.nn.softmax(scores, axis=-1)
        weights = self.dropout(weights)

        context = weights @ value
        context = context.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_length, self.num_query_heads * self.head_dim
        )

        return self.out_nn(context)


class LayerNorm(nnx.Module):
    def __init__(self, embed_dim: int, eps: float = 1e-5) -> None:
        self.eps = eps
        self.scale = nnx.Param(jnp.ones(embed_dim))
        self.shift = nnx.Param(jnp.zeros(embed_dim))

    def __call__(self, x: jax.Array) -> jax.Array:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True, correction=0)

        norm_x = (x - mean) / jnp.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nnx.Module):
    def __call__(self, x: jax.Array) -> jax.Array:
        return (
            0.5
            * x
            * (1 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * jnp.pow(x, 3))))
        )


class FeedForward(nnx.Module):
    def __init__(self, emb_dim: int, rngs: nnx.Rngs, emb_dim_multiply: int = 4) -> None:
        self.nn = nnx.Sequential(
            nnx.Linear(emb_dim, emb_dim * emb_dim_multiply, rngs=rngs),
            GELU(),
            nnx.Linear(emb_dim * emb_dim_multiply, emb_dim, rngs=rngs),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.nn(x)


class TransformerBlock(nnx.Module):
    def __init__(
        self,
        embed_dim: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dropout_rate: float,
        seq_length: int,
        emb_dim_multiply: int,
        rngs: nnx.Rngs,
    ) -> None:
        self.attention = GroupQueryAttention(
            embed_dim=embed_dim,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dropout_rate=dropout_rate,
            seq_length=seq_length,
            rngs=rngs
        )

        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, emb_dim_multiply=emb_dim_multiply, rngs=rngs)
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:

        shortcut = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + shortcut

        return x


class GPTModel(nnx.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_dim: int,
        seq_length: int,
        dropout_rate: float,
        n_layers: int,
        emb_dim_multiply: int,
        rngs: nnx.Rngs,
    ) -> None:
        self.tok_embed = nnx.Embed(vocab_size, embed_dim, rngs=rngs)
        self.pos_embed = nnx.Embed(seq_length, embed_dim, rngs=rngs)
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

        self.blocks = nnx.Sequential(
            *[
                TransformerBlock(
                    embed_dim,
                    num_query_heads,
                    num_kv_heads,
                    head_dim,
                    dropout_rate,
                    seq_length,
                    emb_dim_multiply,
                    rngs=rngs,
                )
                for _ in range(n_layers)
            ]
        )

        self.final_norm = LayerNorm(embed_dim)
        self.out_nn = nnx.Linear(embed_dim, vocab_size, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        batch_size, seq_length = x.shape

        tok_emb = self.tok_embed(x)
        pos_emb = self.pos_embed(jnp.arange(seq_length))

        x = tok_emb + pos_emb
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.final_norm(x)
        return self.out_nn(x)
    
def generate_text_simple(model: GPTModel, idx: jax.Array, max_new_tokens: int, context_size: int) -> str:
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        logits = model(idx_cond)

        logits = logits[:, -1, :]
        probs = jax.nn.softmax(logits, axis=-1)
        idx_next = jnp.argmax(probs, axis=-1, keepdims=True)
        idx = jnp.concatenate([idx, idx_next], axis=-1)

    return idx

if __name__ == "__main__":
    import jax.random as random

    tokenizer = tiktoken.get_encoding("gpt2")

    start_context = "Hello, I am"
    tokenized_context = tokenizer.encode(start_context)
    tokenized_context = jnp.array(tokenized_context)[None, :]

    print("Tokenized Context:", tokenized_context)
    print("Tokenized Context Shape:", tokenized_context.shape)

    rng = nnx.Rngs(0)
    model = GPTModel(
        vocab_size=tokenizer.n_vocab,
        embed_dim=512,
        num_query_heads=8,
        num_kv_heads=4,
        head_dim=64,
        seq_length=tokenized_context.shape[1],
        dropout_rate=0.1,
        n_layers=6,
        emb_dim_multiply=4,
        rngs=rng,
    )
    output = generate_text_simple(model, tokenized_context, max_new_tokens=6, context_size=tokenized_context.shape[1])

    print("Output:", output)
    print("Output Shape:", output.shape)

    decoded_text = tokenizer.decode(output.squeeze(0).tolist())
    print("Decoded Text:", decoded_text)