from mle import *
from generate_ngram import generate_ngrams
import math

# assume log product
def compute_perplexity(prob, N):
    # print("prob: %f"%(prob))

    perplexity = math.exp((-1.0 / N) * prob)
    return perplexity


def test_compute_perplexity():

    probabilities = [0.1] * 10
    # print probabilities
    logProduct = reduce((lambda x, y: x + y), map(lambda x: math.log(x), filter(lambda x: x > 0, probabilities)))
    # print logProduct
    perplexity = compute_perplexity(logProduct, 10, True)
    print "perplexity of textbook example: %f"%(perplexity)
    exit(0)
    infile = open('1.tra', 'r')

    n = 2
    corpus = infile.readlines()
    tokens = [item for sublist in corpus for item in sublist]
    model = generate_model(n, tokens)

    devfile = open('1.dev', 'r')
    dev_corpus = devfile.readlines()
    dev_tokens = [item for sublist in corpus for item in sublist]
    ngrams = list(generate_ngrams(dev_tokens, model.n))

    probabilities = [compute_mle_laplace(ngram, model) for ngram in ngrams]
    # print probabilities
    logProduct = reduce((lambda x, y: x + y), map(lambda x: math.log(x), filter(lambda x: x > 0, probabilities)))
    print compute_perplexity(logProduct, len(ngrams), True)

    # model = generate_model(n, tokens)
def reduce_to_log_prob(probabilities):
    logProduct = reduce((lambda x, y: x + y), map(lambda x: math.log(x), map(lambda x: 0.001 if x == 0 else x, probabilities)))
    return logProduct
