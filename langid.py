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
    for file in filter(lambda x: x.endswith('tra'), training_files):
        print('Generating model for %s ...'%(file))
        infile = open(training_dir + file, 'r')

        corpus = infile.readlines()
        tokens = [item for sublist in corpus for item in sublist]
        model = generate_model(n, tokens)
        models_dict.update({file : model})

    print ("dictionary size %d"%(len(models_dict)))
    for file in filter(lambda x: x.endswith('dev'), dev_files)[0:1]:
        print('computing model for %s ...'%(file))

        infile = open(dev_dir + file, 'r')

        corpus = infile.readlines()
        tokens = [item for sublist in corpus for item in sublist]
        perplexity_dict = dict()
        for file_name, model in models_dict.iteritems():
            ngrams = generate_ngrams(tokens, model.n)
            probabilities = [compute_mle(ngram, model) for ngram in ngrams]
            logProduct = reduce((lambda x, y: x + y), map(lambda x: math.log(x), filter(lambda x: x >0, probabilities)))
            # print filter(lambda x: x > 0, probabilities)
            # print("log product %f"%(logProduct))
            doc_prob = math.exp(logProduct)
            # print("prob %f"%(doc_prob))
            # print("vocabulary size%d"%(len(model.tokens)))
            perplexity = doc_prob ** (-1.0/len(model.tokens))
            # print("perplexity %f"%(perplexity))
            perplexity_dict.update({file_name: perplexity})
        print perplexity_dict
        perplexities = perplexity_dict.values()
        perplexities.sort()


        #print perplexities[0]
        #print perplexity_dict['udhr-als.txt.tra']





if __name__ == "__main__": main()
