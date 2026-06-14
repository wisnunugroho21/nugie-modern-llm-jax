import math
from typing import Any

import flax.nnx as nnx
import grain.python as grain
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from datasets import Dataset, load_dataset
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
            rngs=rngs,
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


class HuggingFaceDataSource(grain.RandomAccessDataSource):
    def __init__(self, hf_ds: Dataset) -> None:
        self.hf_ds = hf_ds

    def __len__(self) -> int:
        return len(self.hf_ds)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.hf_ds[index]


class TokenizerAndShift(grain.MapTransform):
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 128) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def map(self, element: dict[str, Any]) -> dict[str, Any]:
        encoded: dict[str, list[int]] = self.tokenizer(
            element["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors=None,
        )

        tokens: list[int] = encoded["input_ids"]

        new_element = {
            "inputs": tokens[:-1],
            "targets": tokens[1:],
            "attention_mask": encoded["attention_mask"][:-1],
        }

        return new_element


class ConvertToJaxArrays(grain.MapTransform):
    def map(self, element: dict[str, Any]) -> dict[str, Any]:
        for key in ["inputs", "targets", "attention_mask"]:
            element[key] = jnp.array(np.array(element[key]))
        return element


class FilterEmptyLines(grain.FilterTransform):
    def filter(self, element: dict[str, Any]) -> bool:
        return len(element["text"].strip()) > 0


def build_dataloader(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 8,
    max_length: int = 128,
) -> grain.DataLoader:
    source = HuggingFaceDataSource(dataset)

    sampler = grain.IndexSampler(
        num_records=len(source),
        num_epochs=1,
        shard_options=grain.ShardOptions(
            shard_index=0, shard_count=1, drop_remainder=True
        ),
        shuffle=True,
        seed=42,
    )

    loader = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=[
            FilterEmptyLines(),
            TokenizerAndShift(tokenizer, max_length=max_length),
            ConvertToJaxArrays(),
            grain.Batch(batch_size=batch_size, drop_remainder=True),
        ],
        worker_count=2,
    )

    return loader


def loss_fn(model: nnx.Module, batch: dict[str, jax.Array]) -> jax.Array:
    logits = model(batch["inputs"])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch["targets"]
    ).mean()

    return loss


@nnx.jit
def train_step(
    model: nnx.Module, optimizer: nnx.Optimizer, batch: dict[str, jax.Array]
) -> jax.Array:
    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model, batch)

    optimizer.update(model, grads)
    return loss


@nnx.jit
def eval_step(model: nnx.Module, batch: dict[str, jax.Array]) -> jax.Array:
    logits = model(batch["inputs"])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch["targets"]
    ).mean()

    return loss

def save_checkpoint(
    mngr: ocp.CheckpointManager,
    step: int, 
    model: nnx.Module, 
    optimizer: nnx.Optimizer, 
    data_iterator
) -> None:
    """Bundles model weights, optimizer momentum, and Grain iterator state to disk."""
    print(f"Saving checkpoint at step {step}...")
    
    # 1. Get the NNX state (Model + Optimizer)
    _, nnx_state = nnx.split((model, optimizer))
    
    # 2. Get the Grain iterator state
    grain_state = data_iterator.get_state()
    
    # 3. Create a unified PyTree dictionary
    unified_state = {
        "nnx": nnx_state,
        "grain": grain_state
    }
    
    # 4. Save the unified state
    mngr.save(
        step, 
        args=ocp.args.StandardSave(unified_state)
    )
    
    # Block until save is complete to ensure safety
    mngr.wait_until_finished()
    print("Save complete!")
    

def restore_checkpoint(
    mngr: ocp.CheckpointManager,
    model: nnx.Module, 
    optimizer: nnx.Optimizer, 
    data_iterator
) -> int:
    """Restores the unified state and injects it back into the objects."""
    
    latest_step = mngr.latest_step()
    if latest_step is None:
        print("No existing checkpoints found. Starting from scratch (Step 0).")
        return 0 
    
    print(f"Found checkpoint at step {latest_step}. Restoring...")
    
    # 1. Create the abstract template for NNX
    _, abstract_nnx_state = nnx.split((model, optimizer))
    
    # 2. Create the abstract template for Grain 
    # (We can just use its current empty state as the template)
    abstract_grain_state = data_iterator.get_state()
    
    # 3. Assemble the abstract unified state
    abstract_unified_state = {
        "nnx": abstract_nnx_state,
        "grain": abstract_grain_state
    }
    
    # 4. Restore from disk
    restored_state = mngr.restore(
        latest_step, 
        args=ocp.args.StandardRestore(abstract_unified_state)
    )
    
    # 5. Inject the restored states back into their respective objects
    nnx.update((model, optimizer), restored_state["nnx"])
    data_iterator.set_state(restored_state["grain"])
    
    print("Restore complete! Model, Optimizer, and DataLoader are synchronized.")
    return latest_step


def train_and_evaluate(num_epochs: int = 1, eval_every_n_steps: int = 5):
    train_dataset: Dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    val_dataset: Dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")

    gpt2tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained("gpt2")
    gpt2tokenizer.pad_token = gpt2tokenizer.eos_token

    train_loader: grain.DataLoader = build_dataloader(
        train_dataset, gpt2tokenizer, batch_size=8, max_length=128
    )
    val_loader: grain.DataLoader = build_dataloader(
        val_dataset, gpt2tokenizer, batch_size=8, max_length=128
    )

    rngs: nnx.Rngs = nnx.Rngs(0)
    model: nnx.Module = GPTModel(
        vocab_size=gpt2tokenizer.vocab_size,
        embed_dim=512,
        num_query_heads=8,
        num_kv_heads=4,
        head_dim=64,
        seq_length=127,
        dropout_rate=0.1,
        n_layers=6,
        emb_dim_multiply=4,
        rngs=rngs,
    )

    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=3e-4), wrt=nnx.Param)

    step = 0
    print("Starting training...")

    for epoch in range(num_epochs):
        for batch in train_loader:
            train_loss = train_step(model, optimizer, batch)

            if step % eval_every_n_steps == 0 and step > 0:
                total_val_loss = 0.0
                val_steps = 0

                for val_batch in val_loader:
                    val_loss = eval_step(model, val_batch)
                    total_val_loss += val_loss
                    val_steps += 1

                avg_val_loss = total_val_loss / val_steps
                perplexity = math.exp(avg_val_loss)

                print(
                    f"Val Loss: {avg_val_loss:.4f} | Perplexity: {perplexity:.2f} | Epoch: {epoch + 1}/{num_epochs}"
                )

            print(
                f"Step {step:04d} | Train Loss: {train_loss:.4f} | Epoch: {epoch + 1}/{num_epochs}"
            )
            step += 1

        step = 0  # Reset step count after each epoch


if __name__ == "__main__":
    import sys

    from absl import flags

    flags.FLAGS(sys.argv)  # Parse flags to keep Grain multiprocessing happy

    train_and_evaluate()
