import string
from collections import Counter
from nltk import tokenize

class Tokenizer:
    def __init__(self):
        pass

    def tokenize_doc(self, doc):
        bow = Counter()
        tokens = filter(self.filter_tokens, tokenize.word_tokenize(doc))
        tokens = map(self.normalize_tokens, tokens)
        for token in tokens:
            bow[token] += 1
        return bow    

    def filter_tokens(self, token):
        if token in string.punctuation:
            return False
        
        for char in token:
            if char.isdigit():
                return False

        return True

    def normalize_tokens(self, token):
        return token.lower()

t = Tokenizer()
print(t.tokenize_doc("Hello, my name is 384."))