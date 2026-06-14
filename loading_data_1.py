import flax.nnx as nnx
import grain.python as grain
import jax
import jax.numpy as jnp
import numpy as np
from tiktoken import Encoding


class GPTDatasetV1(grain.RandomAccessDataSource):
    def __init__(
        self, txt: str, tokenizer: Encoding, max_length: int, stride: int
    ) -> None:
        self.input_ids: list[jax.Array] = []
        self.target_ids: list[jax.Array] = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]

            input_array = jnp.array(np.array(input_chunk))
            target_array = jnp.array(np.array(target_chunk))

            self.input_ids.append(input_array)
            self.target_ids.append(target_array)

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int) -> tuple[jax.Array, jax.Array]:
        return self.input_ids[index], self.target_ids[index]
