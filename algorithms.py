import math
import collections
import nltk
import time

START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
MINUS_INFINITY_SENTENCE_LOG_PROB = -1000

# Calculates unigram, bigram, and trigram probabilities given a training corpus
def calc_probabilities(training_corpus):
    unigram_c = collections.defaultdict(int)
    bigram_c = collections.defaultdict(int)
    trigram_c = collections.defaultdict(int)


    for sentence in training_corpus:
        tokens0 = sentence.strip().split()
        tokens1 = tokens0 + [STOP_SYMBOL]
        tokens2 = [START_SYMBOL] + tokens0 + [STOP_SYMBOL]
        tokens3 = [START_SYMBOL] + [START_SYMBOL] + tokens0 + [STOP_SYMBOL]
        # unigrams
        for unigram in tokens1:
            unigram_c[unigram] += 1

        # bigrams
        for bigram in nltk.bigrams(tokens2):

            bigram_c[bigram] += 1

        # trigrams
        for trigram in nltk.trigrams(tokens3):
            trigram_c[trigram] += 1

    unigrams_len = sum(unigram_c.itervalues())
    unigram_p = {k: math.log(float(v) / unigrams_len, 2) for k, v in unigram_c.iteritems()}

    # calc P(W2|W1) = P(W2,W1) / P(W1) = C(W2,W1) / C(W1)
    unigram_c[START_SYMBOL] = len(training_corpus)
    bigram_p = {k: math.log(float(v) / unigram_c[k[0]], 2) for k, v in bigram_c.iteritems()}

    bigram_c[(START_SYMBOL, START_SYMBOL)] = len(training_corpus)
    trigram_p = {k: math.log(float(v) / bigram_c[k[:2]], 2) for k, v in trigram_c.iteritems()}
    return unigram_p, bigram_p, trigram_p

# Calculates scores (log probabilities) for every sentence
def score(ngram_p, n, corpus):
    scores = []
    for sentence in corpus:
        sentence_score = 0
        tokens0 = sentence.strip().split()
        if n == 1:
            tokens = tokens0 + [STOP_SYMBOL]
        elif n == 2:
            tokens = nltk.bigrams([START_SYMBOL] + tokens0 + [STOP_SYMBOL])
        elif n == 3:
            tokens = nltk.trigrams([START_SYMBOL] + [START_SYMBOL] + tokens0 + [STOP_SYMBOL])
        else:
            raise ValueError('Parameter "n" has an invalid value %s' % n)
        for token in tokens:
            try:
                p = ngram_p[token]
            except KeyError:
                p = MINUS_INFINITY_SENTENCE_LOG_PROB
            sentence_score += p
        scores.append(sentence_score)
    return scores


# Calculates scores (log probabilities) for every sentence with a linearly interpolated model
def linearscore(unigrams, bigrams, trigrams, corpus):
    """Linear interpolate the probabilities."""
    scores = []
    # Set lambda equal to all the n-grams so that it sums up to 1.
    lambda_ = 1.0 / 3
    for sentence in corpus:
        interpolated_score = 0
        tokens0 = sentence.strip().split()
        for trigram in nltk.trigrams([START_SYMBOL] + [START_SYMBOL] + tokens0 + [STOP_SYMBOL]):
            try:
                p3 = trigrams[trigram]
            except KeyError:
                p3 = MINUS_INFINITY_SENTENCE_LOG_PROB
            try:
                p2 = bigrams[trigram[1:3]]
            except KeyError:
                p2 = MINUS_INFINITY_SENTENCE_LOG_PROB
            try:
                p1 = unigrams[trigram[2]]
            except KeyError:
                p1 = MINUS_INFINITY_SENTENCE_LOG_PROB
            interpolated_score += math.log(lambda_ * (2 ** p3) + lambda_ * (2 ** p2) + lambda_ * (2 ** p1), 2)
        scores.append(interpolated_score)
    return scores

# Replace space with '_', add space between characters for each word
def pre_process(corpus):
    corpus_updated = ''
    for sentence in corpus:
        sentence_converted = ''
        words = sentence.strip().split()
        for word in words:
            sentence_converted += ' '.join(word)
            sentence_converted += ' _ '
        sentence_converted = sentence_converted[:-3]+'\n'
        corpus_updated += sentence_converted
    return corpus_updated.strip()

DATA_PATH = '811_a1_train/'
DEV_PATH = '811_a1_dev/'
OUTPUT_PATH = 'output/'


def process(train_file):
    # get data
    infile = open(DATA_PATH + train_file, 'r')
    corpus = infile.readlines()
    corpus = pre_process(corpus)
    #print corpus
    infile.close()

    # calculate ngram probabilities
    unigrams, bigrams, trigrams = calc_probabilities(corpus)

    return unigrams,bigrams,trigrams


def calculate(test_file,unigrams,bigrams,trigrams):
    # open test file
    infile = open(DEV_PATH + test_file,'r')
    sample1 = infile.readlines()
    sample1 = pre_process(sample1)
    infile.close()

    # linear interpolation
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)

    # calculate perplexity
    M = 0
    for sentence in sample1:
        words = sentence.split()
        M += len(words) + 1

    perplexity = 0
    for score in sample1scores:
       perplexity += float(score)  # assume log probability

    perplexity /= M
    perplexity = 2 ** (-1 * perplexity)
    return perplexity


def main():
    unigrams,bigrams,trigrams = process('udhr-eng.txt.tra')
    pp = calculate('udhr-deu_1996.txt.dev',unigrams,bigrams,trigrams)
    print pp

if __name__ == "__main__": main()
