from mle import *
from generate_ngram import generate_ngrams
from perplexity import *

import os
import math
import time


def main():


    method_arg = sys.argv[1]
    training_dir = sys.argv[2]
    test_dir = sys.argv[3]

    n = 2
    if method_arg == "--unsmoothed":
        mle_method = compute_mle
        n = 3
    elif method_arg == "--laplace":
        mle_method = compute_mle_laplace
        n = 3
    elif method_arg == "--interpolation":
        mle_method = compute_mle_interpolation
        n = 4
    else:
        raise ValueError("Invalid option, must be one of --unsmoothed --laplace --interpolation")
    #
    # training_dir = '811_a1_train/'
    # # dev_dir = '811_a1_dev/'
    # dev_dir = '811_a1_dev/'
    identify_language(mle_method, n, training_dir, test_dir)

def get_method(method_arg):
    if method_arg == "--unsmoothed":
        mle_method = compute_mle
    elif method_arg == "--laplace":
        mle_method = compute_mle_laplace
    elif method_arg == "--interpolation":
        mle_method = compute_mle_interpolation
    else:
        raise ValueError("Invalid option, must be one of --unsmoothed --laplace --interpolation")
    return mle_method

def identify_language(mle_method, n, training_dir, dev_dir):



    training_files = os.listdir(training_dir)
    dev_files = os.listdir(dev_dir)
    dev_files.sort()
    models_dict = dict()
    training_start_time = time.time()
    for file in filter(lambda x: x.endswith('.tra'), training_files):

    # for file in filter(lambda x: x.endswith('eng.txt.tra') or x.endswith('xho.txt.tra'), training_files):
        # print('Generating model for %s ...'%(file))
        infile = open(training_dir + file, 'r')

        corpus = infile.readlines()
        tokens = [item for sublist in corpus for item in sublist]
        model = generate_model(n, tokens)
        models_dict.update({file : model})

    # print ("dictionary size %d"%(len(models_dict)))
    num_correct = 0
    test_start_time = time.time()
    for test_file in dev_files:
        # print('computing model for %s ...'%(file))
        file_start_time = time.time()
        infile = open(dev_dir + test_file, 'r')

        corpus = infile.readlines()
        tokens = [item for sublist in corpus for item in sublist]
        perplexities = []

        for model_name, model in models_dict.iteritems():

            # print(" testing model: " + file_name)
            ngrams = list(generate_ngrams(tokens, model.n))
            # print("length of tokens %d"%(len(ngrams)))
            probabilities = [mle_method(ngram, model) for ngram in ngrams]
            map_zeros = True
            if 0 in probabilities and not map_zeros:
                perplexity = float('nan')
            else:
                if map_zeros:
                    logProduct = reduce((lambda x, y: x + y), map(lambda x: math.log(x), map(lambda x: 0.0000001 if x == 0 else x, probabilities)))
                else:
                    logProduct = reduce((lambda x, y: x + y), map(lambda x: math.log(x), probabilities))

                perplexity = compute_perplexity(logProduct, len(ngrams))
            perplexities.append((model_name, perplexity))


        best_match = min(perplexities, key=lambda x:x[1])

        if best_match[0].split('txt.tra')[0] == test_file.split('txt.dev')[0]:
            num_correct += 1

        # print "%f s to process %s"%(time.time()-file_start_time, test_file)

        # print ("lowest perplexity value: %s"%(str(min(perplexities, key=lambda x:x[1]))))
        print ("%s %s %f %d"%(test_file, best_match[0], best_match[1], models_dict.values()[0].n ))
        # print perplexity_dict['udhr-als.txt.tra']
    # print "number correct: %d"%(num_correct)
    finish_time = time.time()
    total_time_elapsed = finish_time - training_start_time
    training_time = test_start_time - training_start_time
    test_time = finish_time - test_start_time
    # print("Training time: %f Test time: %f Total Time %f"%(training_time, test_time, total_time_elapsed))


if __name__ == "__main__": main()
