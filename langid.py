from input import *
from generate_ngram import generate_ngrams

import os
import math


def main():
    n = 2
    training_dir = '811_a1_train/'
    dev_dir = '811_a1_dev/'
    training_files = os.listdir(training_dir)
    dev_files = os.listdir(dev_dir)
    models_dict = dict()
    for file in filter(lambda x: x.endswith('.tra'), training_files):
        # print('Generating model for %s ...'%(file))
        infile = open(training_dir + file, 'r')

        corpus = infile.readlines()
        tokens = [item for sublist in corpus for item in sublist]
        model = generate_model(n, tokens)
        models_dict.update({file : model})

    print ("dictionary size %d"%(len(models_dict)))
    for file in filter(lambda x: x.endswith('eng.txt.dev') , dev_files)[0:1]:
        print('computing model for %s ...'%(file))

        infile = open(dev_dir + file, 'r')

        corpus = infile.readlines()
        tokens = [item for sublist in corpus for item in sublist]
        prob_tuples = []
        perplexities = []
        for file_name, model in models_dict.iteritems():
            ngrams = generate_ngrams(tokens, model.n)
            probabilities = [compute_mle(ngram, model) for ngram in ngrams]
            print probabilities
            logProduct = reduce((lambda x, y: x + y), map(lambda x: math.log(x), filter(lambda x: x > 0, probabilities)))
            # print filter(lambda x: x > 0, probabilities)
            print("--------------------%s-----------------"%(file_name))
            print("log product %f"%(logProduct))
            doc_prob = math.exp(logProduct)
            print("prob %f"%(doc_prob))
            print("len %d"%(len(tokens)))
            prob_tuples.append((file_name, doc_prob))
            perplexity = doc_prob ** (-1.0/len(tokens))
            print("perplexity %f"%(perplexity))
            print("---------------------------------------")
            perplexities.append((file_name, perplexity))
        prob_tuples.sort(key=lambda x:x[1])
        # print prob_tuples

        perplexities.sort(key=lambda x:x[1])

        print ("lowest prob value: %s"%(str(prob_tuples[0])))
        print("highest prob value: %s"%(str(prob_tuples[len(prob_tuples)-1])))

        print ("lowest perplexity value: %s"%(str(perplexities[0])))
        # print perplexity_dict['udhr-als.txt.tra']


if __name__ == "__main__": main()
