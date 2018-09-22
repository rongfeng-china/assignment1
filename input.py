
from generate_ngram import generate_ngrams
import nltk
import sys

def main():
    method = sys.argv[1]
    infile = open('811_a1_train/' + 'udhr-eng.txt.tra', 'r')
    # infile = open('sentence.tra', 'r')


    corpus = infile.readlines()
    char_array = [item for sublist in corpus for item in sublist]
    n = 2
    n_grams = generate_ngrams(char_array, n)
    nprime_grams = generate_ngrams(char_array, n-1)
    freqDist = nltk.FreqDist(n_grams)
    nprime_freqDist = nltk.FreqDist(nprime_grams)
    print "correct value is 0.0425848295499"
    print compute_mle(('e', 'a'), freqDist, nprime_freqDist, char_array)
    #
    #
    #
    # if method == "--unsmoothed":
    #     ngram_list = list(n_grams)
    #     # vocab_count = len(set(char_array))
    #     # ngram_count = ngram_list.count(ngram)
    #     # ngram_count / len(ngram_list)
    #     # freqDist = nltk.FreqDist(n_grams)
    #     # print freqDist.freq(('e', 'a'))
    #     # print freqDist.freq(('a', 'e'))
    #     # ngram_count = ngram_list.count(ngram)
    #     # ngram_count / len(ngram_list)
    #     # print ngram_list.count(('e', 'a')) * 1.0 / char_array.count('a')
    #
    # elif method == "--laplace":
    #     freqDist = nltk.FreqDist(n_grams)
    #     # freqDist = nltk.LaplaceProbDist(freqDist)
    #     retrun freqDist;
    # # elif method == "--interpolation":
    # #     # freqDist = interpolation()
    # else:
    #     raise ValueError("Invalid smoothing method: " + method)
    # #

def compute_mle(ngram, ngram_freqDist, nprime_freqDist, tokens):

    #TODO handle the case for unigrams

    print ngram[0]
    print(tokens.count(ngram[0]))
    print(len(tokens))
    ngram_prime = ngram[0:len(ngram)-1]
    print "ngram " + str(ngram)

    print "ngram prime " + str(ngram_prime)
    print ngram_freqDist.freq(ngram)
    print nprime_freqDist.freq(ngram_prime)
    print len(tokens)
    mle = ngram_freqDist.freq(ngram) / nprime_freqDist.freq(ngram_prime)
    return mle

#
# def laplace():
#     freqDist = nlt.FreqDist(n_grams)



if __name__ == "__main__": main()
