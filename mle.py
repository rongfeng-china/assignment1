
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
        self._all_grams = None
        self._all_freqs = None


    @property
    def normalized_lambdas(self):
        if not self._normalized_lambdas:
            lambdas = calc_lambdas(self)
            self._normalized_lambdas = calc_normalized_lambdas(lambdas)

        return self._normalized_lambdas

    @property
    def all_freqs(self):
        if not self._all_freqs:
            all_freqs = map(lambda x: nltk.FreqDist(x), self.all_grams)
            self._all_freqs = all_freqs
        return self._all_freqs

    @property
    def all_grams(self):
        if not self._all_grams:
            all_grams = [None] * self.n
            for i in range(0, self.n):

                if i == 0:
                    all_grams[i] = self.n_grams
                if i == 1:
                    all_grams[i] = self.nprime_grams

                all_grams[i] = list(generate_ngrams(self.tokens, self.n-i))
            self._all_grams = all_grams

        return self._all_grams




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



    # print model.normalized_lambdas
    weighted_mles = compute_weighted_mles(weighted_mles, model.normalized_lambdas, ngram, model)
    # print "MLE: %f"%(sum(weighted_mles))
    return sum(weighted_mles)

def calc_normalized_lambdas(lambdas):
    lambda_sum = sum(lambdas)
    normalized_lambdas = [float(i)/lambda_sum for i in lambdas]
    return normalized_lambdas

def compute_weighted_mles(lambdas, ngram, model):

    # for x in lambdas:
    #
    #     mle = compute_mle(ngram, generate_model(len(ngram), model.tokens)) * x
    #     mles.append(mle)

    all_freqs = model.all_freqs


    weighted_mles = map(lambda x: compute_mle(ngram, all_freqs[len(lambdas) -len(ngram)],
                                                all_freqs[len(lambdas)-len(ngram)-1], model))
    # if len(ngram) == 1:
    #     mle = compute_mle(ngram, generate_model(len(ngram), model.tokens)) * lambdas[len(lambdas)-1]
    #
    #     # print "MLE of %s: %f"%(str(ngram), mle)
    #
    #     return mles.append(mle)
    #
    # mle = compute_mle(ngram, generate_model(len(ngram), model.tokens)) * lambdas[len(lambdas) - len(ngram)]
    #
    # # print "MLE of %s: %f"%(str(ngram), mle)
    # mles.append(mle)
    # ngram_prime = ngram[0:len(ngram)-1]
    # compute_weighted_mles(mles, lambdas, ngram_prime, model)
    #
    # return mles


def calc_lambdas(model):
    print "calc lambdas called..."
    # print model.all_grams
    # print model.all_freqs
    start_time = time.time()
    lambdas = [0] * model.n
    for gram in model.n_grams:


        frequencies = calc_max_freq(gram, model)
        index = frequencies.index(max(frequencies, key = lambda x: x[0]))
        lambdas[index] += + max(frequencies, key = lambda x: x[0])[1]

    print "time to calc lambdas: %f"%(time.time()-start_time)
    return lambdas

def calc_max_freq(starting_ngram, model):
    ngram = starting_ngram
    all_freqs = model.all_freqs
    frequencies = []
    while len(ngram) > 1:
        ngram_prime = ngram[1:len(ngram)]
        ngram_freqDist = all_freqs[len(all_freqs)-len(ngram)]
        ngram_prime_freqDist = all_freqs[len(all_freqs)-len(ngram_prime)]

        ngram_count = ngram_freqDist.freq(ngram) *ngram_freqDist.N()
        ngram_prime_count = ngram_prime_freqDist.freq(ngram_prime) * ngram_prime_freqDist.N()

        if ngram_prime_count == 1:
            frequencies.append((0, ngram_count))
        else:
            freq = (ngram_count - 1.0) / (ngram_prime_count -1)
            frequencies.append((freq, ngram_count))
        ngram = ngram_prime

    if len(ngram) == 1:
        ngram_freqDist = all_freqs[len(all_freqs)-len(ngram)]
        ngram_count = ngram_freqDist.freq(ngram) *ngram_freqDist.N()
        freq = (ngram_count -1.0)/ (len(model.tokens) -1)
        frequencies.append((freq, ngram_count))

    return frequencies



# def calc_max_freq(frequencies, starting_ngram, tokens):
#     ngram = starting_ngram
#     start_time = time.time()
#     ngrams = list(generate_ngrams(tokens, len(ngram)))
#     ngram_freq_dist = nltk.FreqDist(ngrams)
#
#     while len(ngram) > 1:
#
#         ngram_prime = ngram[1:len(ngram)]
#         ngrams_prime= list(generate_ngrams(tokens, len(ngram_prime)))
#         # nprime_freq_dist = nltk.FreqDist(ngrams)
#
#         ngram_prime_count = ngrams_prime.count(ngram_prime)
#         if ngram_prime_count == 1:
#             frequencies.append((0, ngrams.count(ngram)))
#         else:
#
#             freq = (ngrams.count(ngram) - 1.0) /  (ngram_prime_count -1)
#             frequencies.append((freq, ngrams.count(ngram)))
#
#
#         ngram = ngram_prime
#
#         ngrams = ngrams_prime
#
#         # print "iteration %d: %f s elapsed since start"%(i, time.time()-start_time)
#
#     if len(ngram) == 1:
#
#         ngrams = list(generate_ngrams(tokens, 1))
#         freq = (1.0 * ngrams.count(ngram) -1)/ (len(tokens) -1)
#         return frequencies.append((freq, ngrams.count(ngram)))
#
#
#
#
#
#     # ngrams = list(generate_ngrams(tokens, len(ngram)))
#     #
#     # ngram_prime = ngram[1:len(ngram)]
#     # ngrams_prime = list(generate_ngrams(tokens, len(ngram_prime)))
#     # ngram_prime_count = ngrams_prime.count(ngram_prime)
#     # if ngram_prime_count == 1:
#     #     frequencies.append((0, ngrams.count(ngram)))
#     #     calc_max_freq(frequencies, ngram_prime, tokens)
#     # else:
#     #
#     #     freq = (ngrams.count(ngram) - 1.0) /  (ngram_prime_count -1)
#     #     frequencies.append((freq, ngrams.count(ngram)))
#     #     calc_max_freq(frequencies, ngram_prime, tokens)


def compute_mle_laplace(ngram, model):
    # print("calling laplace")
    if len(ngram) != model.n:
        raise ValueError('Cannot compute on n-gram with length %d on model with length %d'%(len(ngram), model.n))

    ngram_prime = ngram[0:len(ngram)-1]

    vocab = set(model.tokens)

    n_gram_count = model.ngram_freqDist.freq(ngram) * model.ngram_freqDist.N() + 1
    n_prime_count = model.nprime_freqDist.freq(ngram_prime) * model.nprime_freqDist.N() + len(vocab)
    if len(ngram) == 1:
        n_prime_count = len(vocab) + len(model.tokens)

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
