class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.char_to_id = {}
        self.id_to_char = {}

    def encode(self, text: str) -> list[int]:
        return [self.char_to_id[char] for char in text]

    def decode(self, tokens: list[int]) -> str:
        return ''.join([self.id_to_char[token] for token in tokens]) 
           
    @property
    def vocab_size(self) -> int:
        return len(self.char_to_id)