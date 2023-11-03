import torch


class IntCharTokenizer:
    def __init__(self, text):
        self.chars, self.vocab_size = self._get_uniq_chars(text)
        self.int_to_char = {i: c for i, c in enumerate(self.chars)}
        self.char_to_int = {c: i for i, c in enumerate(self.chars)}

    def _get_uniq_chars(self, text):
        chars = sorted(list(set(text)))
        return chars, len(chars)
    
    def encode(self, text):
        #enc = lambda s: [self.char_to_int[c] for c in s]
        return [self.char_to_int[c] for c in text]
    
    def decode(self, tokens):
        #dec = lambda s: ''.join(self.int_to_char[i] for i in s)
        return ''.join(self.int_to_char[i] for i in tokens)
    