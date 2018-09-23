import math
import collections
import nltk
import time

# Constants to be used by you when you fill the functions
START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
MINUS_INFINITY_SENTENCE_LOG_PROB = -1000

# TODO: IMPLEMENT THIS FUNCTION
# Calculates unigram, bigram, and trigram probabilities given a training corpus
# training_corpus: is a list of the sentences. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function outputs three python dictionaries, where the keys are tuples expressing the ngram and the value is the log probability of that ngram
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

# Prints the output for q1
# Each input is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams, filename):
    # output probabilities
    outfile = open(filename, 'w')

    unigrams_keys = unigrams.keys()
    unigrams_keys.sort()
    for unigram in unigrams_keys:
        outfile.write('UNIGRAM ' + unigram[0] + ' ' + str(unigrams[unigram]) + '\n')

    bigrams_keys = bigrams.keys()
    bigrams_keys.sort()
    for bigram in bigrams_keys:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  + ' ' + str(bigrams[bigram]) + '\n')

    trigrams_keys = trigrams.keys()
    trigrams_keys.sort()    
    for trigram in trigrams_keys:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')

    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence
# ngram_p: python dictionary of probabilities of uni-, bi- and trigrams.
# n: size of the ngram you want to use to compute probabilities
# corpus: list of sentences to score. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function must return a python list of scores, where the first element is the score of the first sentence, etc. 
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

# Outputs a score to a file
# scores: list of scores
# filename: is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence with a linearly interpolated model
# Each ngram argument is a python dictionary where the keys are tuples that express an ngram and the value is the log probability of that ngram
# Like score(), this function returns a python list of scores
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

# DO NOT MODIFY THE MAIN FUNCTION
def process(train_file):
    # get data
    infile = open(DATA_PATH + train_file, 'r') 
    corpus = infile.readlines()
    corpus = pre_process(corpus)
    #print corpus
    infile.close()

    # calculate ngram probabilities 
    unigrams, bigrams, trigrams = calc_probabilities(corpus)
    #q1_output(unigrams, bigrams, trigrams, OUTPUT_PATH + 'A1.txt')

    # score sentences 
    #uniscores = score(unigrams, 1, corpus)
    #biscores = score(bigrams, 2, corpus)
    #triscores = score(trigrams, 3, corpus)
    #score_output(uniscores, OUTPUT_PATH + 'A2.uni.txt')
    #score_output(biscores, OUTPUT_PATH + 'A2.bi.txt')
    #score_output(triscores, OUTPUT_PATH + 'A2.tri.txt')


    #linearscores = linearscore(unigrams, bigrams, trigrams, corpus)
    #score_output(linearscores, OUTPUT_PATH + 'A3.txt')
    return unigrams,bigrams,trigrams


def calculate(test_file,unigrams,bigrams,trigrams):
    # open test file
    infile = open(DEV_PATH + test_file,'r') #'udhr-deu_1996.txt.dev', 'r')
    sample1 = infile.readlines()
    sample1 = pre_process(sample1)
    infile.close()

    # linear interpolation 
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)

    #score_output(sample1scores, OUTPUT_PATH + 'Sample1_scored.txt')
    #score_output(sample2scores, OUTPUT_PATH + 'Sample2_scored.txt')

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
    #print "The perplexity is", perplexity   
    
def main():
    unigrams,bigrams,trigrams = process('udhr-eng.txt.tra')
    pp = calculate('udhr-deu_1996.txt.dev',unigrams,bigrams,trigrams)
    print pp

if __name__ == "__main__": main()
