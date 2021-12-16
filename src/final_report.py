import nltk
import csv
import random

def tokenize_doc(doc):
    sentence_tokens = []
    csv_reader = csv.reader(doc)
    for row in csv_reader:
        sentence_tokens += [nltk.word_tokenize(sentence.lower()) for sentence in nltk.sent_tokenize(row[2])]
    return sentence_tokens

def mask_tokens(sentence_tokens):
    masked_words = []
    for sentence_token in sentence_tokens:
        rand_index = random.randint(0, len(sentence_token) - 1)
        masked_words.append(sentence_token[rand_index])
        sentence_token[rand_index] = '<MASK>'
    return sentence_tokens, masked_words
    
def train_mle(sentence_tokens):
    mle = nltk.lm.MLE(3)
    train, vocab = nltk.lm.preprocessing.padded_everygram_pipeline(3, sentence_tokens)
    mle.fit(train, vocab)
    return mle

def train_laplace(sentence_tokens):
    laplace = nltk.lm.Laplace(3)
    train, vocab = nltk.lm.preprocessing.padded_everygram_pipeline(3, sentence_tokens)
    laplace.fit(train, vocab)
    return laplace

def train_lidstone(sentence_tokens, gamma):
    lidstone = nltk.lm.Lidstone(gamma, 3)
    train, vocab = nltk.lm.preprocessing.padded_everygram_pipeline(3, sentence_tokens)
    lidstone.fit(train, vocab)
    return lidstone

def train_stupid_backoff(sentence_tokens, alpha):
    stupid_backoff = nltk.lm.StupidBackoff(alpha, 3)
    train, vocab = nltk.lm.preprocessing.padded_everygram_pipeline(3, sentence_tokens)
    stupid_backoff.fit(train, vocab)
    return stupid_backoff

def test_model(model, masked_sentence_tokens, masked_words):
    total_correct = 0
    total_word_length = 0
    total_related = 0
    for i, masked_sentence_token in enumerate(masked_sentence_tokens):
        masked_word = masked_words[i]
        j = masked_sentence_token.index('<MASK>')
        max_word = None
        if j > 1:
            prev_word = masked_sentence_token[j - 1]
            prev_prev_word = masked_sentence_token[j - 2]
            max_word = model.generate(1, text_seed=[prev_prev_word, prev_word])
        elif j == 1:
            prev_word = masked_sentence_token[j - 1]
            max_word = model.generate(1, text_seed=[prev_word])
        else:
            max_word = model.generate(1)

        if max_word == masked_word:
            total_word_length += len(max_word)
            total_correct += 1
            total_related += 1
        else:
            for synset in nltk.corpus.wordnet.synsets(masked_word):
                for lemma in synset.lemmas():
                    if lemma.name() == max_word:
                        total_related += 1
                        break

    model_exact_accuracy = (total_correct / len(masked_sentence_tokens)) * 100
    model_relative_accuracy = (total_related / len(masked_sentence_tokens)) * 100
    model_avg_word_length = total_word_length / total_correct
    return model_exact_accuracy, model_relative_accuracy, model_avg_word_length