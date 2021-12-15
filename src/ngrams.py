import nltk
import csv

with open('../data/sample.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    sentence_tokens = []
    print("Tokenizing...")
    for row in csv_reader:
        sentence_tokens += [nltk.word_tokenize(sentence) for sentence in nltk.sent_tokenize(row[2])]
    train, vocab = nltk.lm.preprocessing.padded_everygram_pipeline(3, sentence_tokens)
    lm = nltk.lm.MLE(3)
    print("Training the model...")
    lm.fit(train, vocab)
    print(lm.generate(10))
