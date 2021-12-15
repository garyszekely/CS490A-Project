import string
import csv
import spacy
from collections import Counter
from nltk import tokenize
from spacy.language import Language
from spacy_langdetect import LanguageDetector

class Tokenizer:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        Language.factory('language_detector', func=self.get_lang_detector)
        self.nlp.add_pipe('language_detector', last=True)
        self.total_words = 0

    def get_lang_detector(self, nlp, name):
        return LanguageDetector()

    def tokenize_doc(self, doc):
        doc_lang = self.nlp(doc)._.language

        if doc_lang['language'] == 'en':
            bow = Counter()
            tokens = filter(self.filter_tokens, tokenize.word_tokenize(doc))
            tokens = map(self.normalize_tokens, tokens)
            for token in tokens:
                bow[token] += 1

            return bow
        
        return None

    def filter_tokens(self, token):
        for char in token:
            if char in string.punctuation:
                return False

            if char.isdigit():
                return False
        self.total_words += 1
        return True

    def normalize_tokens(self, token):
        return token.lower()

if __name__ == "__main__":
    t = Tokenizer()
    bow = Counter()
    with open('./data/sample.csv') as file:
        reader = csv.reader(file)
        for row in reader:
            bow.update(t.tokenize_doc(row[2]))

    word_counts = sorted(bow.items(), key=lambda w: w[1], reverse=True)
    word_probs = list(map(lambda w: (w[0], w[1] / t.total_words), word_counts))
    for i in range(30):
        print(word_probs[i])
