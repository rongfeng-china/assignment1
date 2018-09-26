
from generate_ngram import generate_ngrams
import nltk
import sys
import time


class LanguageModel:

    def __init__(self, n, tokens):
        self.n = n
        self.tokens = tokens
        self.n_grams = list(generate_ngrams(tokens, n))
        self.nprime_grams = list(generate_ngrams(tokens, n-1))
        self.ngram_freqDist = nltk.FreqDist(self.n_grams)
        self.nprime_freqDist = nltk.FreqDist(self.nprime_grams)
        self._normalized_lambdas = []


    @property
    def normalized_lambdas(self):
        if not self._normalized_lambdas:
            lambdas = calc_lambdas(self)
            self._normalized_lambdas = calc_normalized_lambdas(lambdas)

        return self._normalized_lambdas


def main():
    method = sys.argv[1]
    infile = open('811_a1_train/' + 'udhr-eng.txt.tra', 'r')
    # infile = open('sentence.tra', 'r')


    corpus = infile.readlines()
    char_array = [item for sublist in corpus for item in sublist]
    # char_array = "the first the second the third the fourth the first".split(' ')
    n = 3
    model = generate_model(n, char_array)


    print compute_mle(('e', 'a', 's'), model)
    print compute_mle_laplace(('e', 'a', 's'), model)


def test():
    method = sys.argv[1]
    infile = open('811_a1_train/' + 'udhr-eng.txt.tra', 'r')
    # infile = open('sentence.tra', 'r')


    corpus = infile.readlines()
    char_array = [item for sublist in corpus for item in sublist]
    # char_array = "the first the second the third the fourth the first".split(' ')
    n = 3
    model = generate_model(n, char_array)

    print compute_mle_interpolation(('e', 'a', 's'), model)

def generate_model(n, tokens):
    # TODO: special case for unigram
    return LanguageModel(n, tokens)

def compute_mle_interpolation(ngram, model):
    if len(ngram) != model.n:
        raise ValueError('Cannot compute on n-gram with length %d on model with length %d'%(len(ngram), model.n))

    if len(ngram) == 1:
        return compute_mle(ngram, model) # should be the same as unsmoothed model

    weighted_mles = []

    # print model.normalized_lambdas
    compute_weighted_mles(weighted_mles, model.normalized_lambdas, ngram, model)

    return sum(weighted_mles)

def calc_normalized_lambdas(lambdas):
    lambda_sum = sum(lambdas)
    normalized_lambdas = [float(i)/lambda_sum for i in lambdas]
    return normalized_lambdas
def compute_weighted_mles(mles, lambdas, ngram, model):

    # if len(lambdas) != (len(ngram) + len(mles)):
    #     raise ValueError("length of lambdas must equal length of ngram %d vs. %d"%(len(lambdas), len(ngram)))
    if len(ngram) == 1:
        mle = compute_mle(ngram, generate_model(len(ngram), model.tokens)) * lambdas[len(lambdas)-1]
        # print "MLE of %s: %f"%(str(ngram), mle)

        return mles.append(mle)

    mle = compute_mle(ngram, generate_model(len(ngram), model.tokens)) * lambdas[len(lambdas) - len(ngram)]

    # print "MLE of %s: %f"%(str(ngram), mle)
    mles.append(mle)
    ngram_prime = ngram[0:len(ngram)-1]
    compute_weighted_mles(mles, lambdas, ngram_prime, model)


def calc_lambdas(model):

    lambdas = [0] * model.n

    for gram in model.n_grams:

        frequencies = []
        calc_max_freq(frequencies, gram, model.tokens)
        index = frequencies.index(max(frequencies, key = lambda x: x[0]))
        lambdas[index] += + max(frequencies, key = lambda x: x[0])[1]


    return lambdas


def calc_max_freq(frequencies, ngram, tokens):
    if len(ngram) == 1:

        ngrams = list(generate_ngrams(tokens, 1))
        freq = (1.0 * ngrams.count(ngram) -1)/ (len(tokens) -1)
        return frequencies.append((freq, ngrams.count(ngram)))
    ngrams = list(generate_ngrams(tokens, len(ngram)))

    ngram_prime = ngram[1:len(ngram)]
    ngrams_prime = list(generate_ngrams(tokens, len(ngram_prime)))
    ngram_prime_count = ngrams_prime.count(ngram_prime)
    if ngram_prime_count == 1:
        frequencies.append((0, ngrams.count(ngram)))
        calc_max_freq(frequencies, ngram_prime, tokens)
    else:

        freq = (ngrams.count(ngram) - 1.0) /  (ngram_prime_count -1)
        frequencies.append((freq, ngrams.count(ngram)))
        calc_max_freq(frequencies, ngram_prime, tokens)


def compute_mle_laplace(ngram, model):
    # print("calling laplace")
    if len(ngram) != model.n:
        raise ValueError('Cannot compute on n-gram with length %d on model with length %d'%(len(ngram), model.n))

    ngram_prime = ngram[0:len(ngram)-1]

    vocab = set(model.tokens)

    # print "ngram freq dist %f"%(model.ngram_freqDist.freq(ngram))
    # print "nprime freq dist %f"%(model.nprime_freqDist.freq(ngram_prime))
    # print len(vocab)
    n_gram_count = model.ngram_freqDist.freq(ngram) * model.ngram_freqDist.N() + 1
    n_prime_count = model.nprime_freqDist.freq(ngram_prime) * model.nprime_freqDist.N() + len(vocab)
    if len(ngram) == 1:
        n_prime_count = len(vocab) + len(model.tokens)
    # print("the n gram: %s"%(str(ngram)))
    # print "ngram count %f"%(n_gram_count)
    # print "nprime count %f"%(n_prime_count)

    mle = n_gram_count / n_prime_count

    return mle

def compute_mle(ngram, model):

    if len(ngram) != model.n:
        print "The ngram%s"%(ngram)
        raise ValueError("Cannot compute on n-gram with length %d on model with length %d"%(len(ngram), model.n))
    if len(ngram) == 1:

        gram_count = float(model.n_grams.count(ngram))
        mle = gram_count/ len(model.tokens)
        return mle
    ngram_prime = ngram[0:len(ngram)-1]
    normalization_factor = ((len(model.tokens)-(len(ngram)-1)) * 1.0 / (len(model.tokens)-(len(ngram_prime)-1)))
    # print "normalization factor" + str(normalization_factor)
    try:
        mle = model.ngram_freqDist.freq(ngram) / model.nprime_freqDist.freq(ngram_prime) * normalization_factor

        return mle
    except ZeroDivisionError:
        return 0.0

if __name__ == "__main__": test()
