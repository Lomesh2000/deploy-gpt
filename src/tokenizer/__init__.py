class BaseTokenizer:

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError("Encode method not implemented.")

    def decode(self, tokens: list[int]) -> str:
        raise NotImplementedError("Decode method not implemented.")

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError("Vocab size property not implemented.")        