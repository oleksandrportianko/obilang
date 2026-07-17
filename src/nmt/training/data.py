"""Tokenized translation dataset and token-budget dynamic batch sampler."""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from nmt.data.records import ParallelRecord
from nmt.tokenization.sentencepiece import TokenizerBundle


@dataclass(frozen=True)
class EncodedExample:
    """Boundary-inclusive source and target token IDs for one aligned row."""

    source_ids: list[int]
    target_ids: list[int]
    source_text: str
    target_text: str
    domain: str | None


class TranslationDataset(Dataset[EncodedExample]):
    """In-memory encoded corpus with maximum-length filtering."""

    def __init__(
        self,
        records: Sequence[ParallelRecord],
        tokenizers: TokenizerBundle,
        maximum_length: int,
    ) -> None:
        """Encode rows once and drop only post-tokenization overlength examples.

        Args:
            records: Validated parallel rows.
            tokenizers: Immutable source/target token IDs.
            maximum_length: Maximum source and target length including BOS/EOS.

        Side effects:
            None outside memory. ``skipped_overlength`` reports rows that passed
            character filtering but exceeded the precise tokenizer limit.
        """
        self.examples: list[EncodedExample] = []
        self.skipped_overlength = 0
        for record in records:
            source_ids = tokenizers.source.encode(record.source)
            target_ids = tokenizers.target.encode(record.target)
            if len(source_ids) > maximum_length or len(target_ids) > maximum_length:
                self.skipped_overlength += 1
                continue
            self.examples.append(
                EncodedExample(
                    source_ids, target_ids, record.source, record.target, record.domain
                )
            )

    def __len__(self) -> int:
        """Return encoded example count."""
        return len(self.examples)

    def __getitem__(self, index: int) -> EncodedExample:
        """Return one encoded example by integer index."""
        return self.examples[index]

    def token_length(self, index: int) -> int:
        """Return padded-token cost proxy for dynamic batch construction."""
        item = self.examples[index]
        return max(len(item.source_ids), len(item.target_ids))


class TokenBatchSampler(Sampler[list[int]]):
    """Build reproducible batches bounded by padded tokens rather than rows."""

    def __init__(
        self,
        dataset: TranslationDataset,
        token_budget: int,
        shuffle: bool,
        seed: int,
        epoch: int = 0,
    ) -> None:
        """Configure length-aware batching.

        A local sort within shuffled pools reduces padding while retaining global
        randomness. The batch cost is ``max_sequence_length * batch_size * 2``
        because both source and target tensors consume memory.
        """
        self.dataset = dataset
        self.token_budget = token_budget
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = epoch

    def set_epoch(self, epoch: int) -> None:
        """Change deterministic shuffle order between epochs."""
        self.epoch = epoch

    def _batches(self) -> list[list[int]]:
        """Materialize one epoch's deterministic index batches."""
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(indices)
            pool_size = 100
            indices = [
                index
                for start in range(0, len(indices), pool_size)
                for index in sorted(
                    indices[start : start + pool_size], key=self.dataset.token_length
                )
            ]
        batches: list[list[int]] = []
        batch: list[int] = []
        longest = 0
        for index in indices:
            item_length = self.dataset.token_length(index)
            proposed_longest = max(longest, item_length)
            proposed_cost = proposed_longest * (len(batch) + 1) * 2
            if batch and proposed_cost > self.token_budget:
                batches.append(batch)
                batch = []
                longest = 0
            batch.append(index)
            longest = max(longest, item_length)
        if batch:
            batches.append(batch)
        if self.shuffle:
            random.Random(self.seed + self.epoch + 1_000_003).shuffle(batches)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        """Yield lists of dataset indices for one epoch."""
        yield from self._batches()

    def __len__(self) -> int:
        """Return exact batch count for the current epoch."""
        return len(self._batches())


@dataclass(frozen=True)
class TranslationBatch:
    """Padded tensors plus original text used for loss and sample reports."""

    source_ids: Tensor
    target_input_ids: Tensor
    target_output_ids: Tensor
    source_texts: list[str]
    target_texts: list[str]

    @property
    def target_tokens(self) -> int:
        """Return number of target labels including padding before loss masking."""
        return self.target_output_ids.numel()


def collate_examples(
    examples: list[EncodedExample], source_pad_id: int, target_pad_id: int
) -> TranslationBatch:
    """Pad variable sequences and shift targets for teacher forcing.

    Returns:
        Source ``[B,S]``, target input ``[B,T-1]`` starting with BOS, and target
        output ``[B,T-1]`` ending with EOS. PAD fills unused batch positions.
    """
    if not examples:
        raise ValueError("Cannot collate an empty translation batch.")
    source_length = max(len(item.source_ids) for item in examples)
    target_length = max(len(item.target_ids) for item in examples)
    source = torch.full((len(examples), source_length), source_pad_id, dtype=torch.long)
    target = torch.full((len(examples), target_length), target_pad_id, dtype=torch.long)
    for row_index, item in enumerate(examples):
        source[row_index, : len(item.source_ids)] = torch.tensor(item.source_ids)
        target[row_index, : len(item.target_ids)] = torch.tensor(item.target_ids)
    return TranslationBatch(
        source,
        target[:, :-1],
        target[:, 1:],
        [item.source_text for item in examples],
        [item.target_text for item in examples],
    )


@dataclass(frozen=True)
class TranslationCollator:
    """Pickle-safe DataLoader collator carrying pair-specific padding IDs."""

    source_pad_id: int
    target_pad_id: int

    def __call__(self, examples: list[EncodedExample]) -> TranslationBatch:
        """Delegate to ``collate_examples`` for one dynamic batch."""
        return collate_examples(examples, self.source_pad_id, self.target_pad_id)
