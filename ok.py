#from input import *
#from generate_ngram import generate_ngrams

import os
import math
from algorithms import process,calculate

def main():
    #n = 3
    training_dir = '811_a1_train/'
    dev_dir = '811_a1_dev/'
    training_files = os.listdir(training_dir)
    dev_files = os.listdir(dev_dir)
    models_dict = dict()
    
    for file in filter(lambda x: x.endswith('tra'), training_files):
        #print('Generating model for %s ...'%(file))
        unigrams, bigrams, trigrams = process(file)
        models_dict.update({file:[unigrams,bigrams,trigrams]})

    #print ("dictionary size %d"%(len(models_dict)))
    for file in filter(lambda x: x.endswith('dev'), dev_files):#[0:1]:
        print('computing model for %s ...'%(file))
        perplexities = dict()
        for train_file in models_dict:
            u,b,t = models_dict[train_file]
            pp = calculate(file,u,b,t)
            perplexities.update({train_file:pp})
        pp_sorted = sorted(perplexities.items(), key=lambda k: k[1])
        print pp_sorted[0]
        print
        


if __name__ == "__main__": main()
