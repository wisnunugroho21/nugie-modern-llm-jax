from typing import Any

import grain.python as grain
import jax.numpy as jnp
import numpy as np
from absl import app
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

hf_dataset: Dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

gpt2tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained("gpt2")
gpt2tokenizer.pad_token = gpt2tokenizer.eos_token


class HuggingFaceDataSource(grain.RandomAccessDataSource):
    def __init__(self, hf_ds: Dataset) -> None:
        self.hf_ds = hf_ds

    def __len__(self) -> int:
        return len(self.hf_ds)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.hf_ds[index]


source = HuggingFaceDataSource(hf_dataset)


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

        element["inputs"] = tokens[:-1]
        element["targets"] = tokens[1:]
        element["attention_mask"] = encoded["attention_mask"][:-1]

        return element


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


def main(argv):
    dataloader: grain.DataLoader = build_dataloader(
        dataset=hf_dataset, tokenizer=gpt2tokenizer, batch_size=8, max_length=128
    )

    for data in dataloader:
        print("Batch Keys:", data.keys())
        print("Inputs Shape:", data["inputs"].shape)
        print("Labels Shape:", data["targets"].shape)

        print("First Input Example:", data["inputs"][0])
        print("First Target Example:", data["targets"][0])
        break


if __name__ == "__main__":
    app.run(main)
