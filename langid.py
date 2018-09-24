from mle import *
from generate_ngram import generate_ngrams

import os
import math


def main():

    method_arg = sys.argv[1]

    if method_arg == "--unsmoothed":
        mle_method = compute_mle
    elif method_arg == "--laplace":
        mle_method = compute_mle_laplace
    elif method_arg == "--interpolation":
        mle_method = compute_mle_interpolation
    else:
        raise ValueError("Invalid option")



    # probabilities = [0.1] * 10
    # print probabilities
    # logProduct = reduce((lambda x, y: x + y), map(lambda x: math.log(x), filter(lambda x: x > 0, probabilities)))
    # print logProduct
    # perplexity = compute_perplexity(logProduct, 10, True)
    # print "perplexity of textbook example: %f"%(perplexity)
    # exit(0)
    n = 3
    training_dir = '811_a1_train/'
    # dev_dir = '811_a1_dev/'
    dev_dir = '811_a1_dev/'

    training_files = os.listdir(training_dir)
    dev_files = os.listdir(dev_dir)
    models_dict = dict()
    for file in filter(lambda x: x.endswith('.tra'), training_files):

    # for file in filter(lambda x: x.endswith('eng.txt.tra') or x.endswith('xho.txt.tra'), training_files):
        # print('Generating model for %s ...'%(file))
        infile = open(training_dir + file, 'r')

        corpus = infile.readlines()
        tokens = [item for sublist in corpus for item in sublist]
        model = generate_model(n, tokens)
        models_dict.update({file : model})

    print ("dictionary size %d"%(len(models_dict)))
    for file in filter(lambda x: x.endswith('.dev') , dev_files):
        print('computing model for %s ...'%(file))

        infile = open(dev_dir + file, 'r')

        corpus = infile.readlines()
        tokens = [item for sublist in corpus for item in sublist]
        perplexities = []

        for file_name, model in models_dict.iteritems():

            # print(" testing model: " + file_name)
            ngrams = list(generate_ngrams(tokens, model.n))
            # print("length of tokens %d"%(len(ngrams)))
            probabilities = [mle_method(ngram, model) for ngram in ngrams]
            # print probabilities
            # print("number of 0s: %d", probabilities.count(0))
            # logProduct = reduce((lambda x, y: x + y), map(lambda x: math.log(x), filter(lambda x: x > 0, probabilities)))
            if probabilities.contains(0):
                perplexity = float('nan')
            else:

                logProduct = reduce((lambda x, y: x + y), map(lambda x: math.log(x), map(lambda x: 0.001 if x == 0 else x, probabilities)))
            # print filter(lambda x: x > 0, probabilities)

                perplexity = compute_perplexity(logProduct, len(ngrams), True)
            # print("--------------------%s-----------------"%(file_name))
            #
            # print("log product %f"%(logProduct))
            # doc_prob = math.exp(logProduct)
            # print("prob %f"%(doc_prob))
            # print("len %d"%(len(tokens)))
            # prob_tuples.append((file_name, doc_prob))
            # # perplexity = doc_prob ** (-1.0/len(tokens))
            # print("perplexity %f"%(perplexity))
            # print("---------------------------------------")
            perplexities.append((file_name, perplexity))



        # print prob_tuples


        # print perplexities
        # print ("lowest prob value: %s"%(str(prob_tuples[0])))
        # print("highest prob value: %s"%(str(prob_tuples[len(prob_tuples)-1])))
        #
        print ("lowest perplexity value: %s"%(str(min(perplexities, key=lambda x:x[1]))))
        # print perplexity_dict['udhr-als.txt.tra']

# assume log product
def compute_perplexity(prob, N, is_logprob):
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


if __name__ == "__main__": main()
