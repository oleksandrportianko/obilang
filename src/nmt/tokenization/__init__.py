"""From-scratch SentencePiece training and immutable tokenizer loading."""

from nmt.tokenization.sentencepiece import TokenizerBundle, train_tokenizer

__all__ = ["TokenizerBundle", "train_tokenizer"]
