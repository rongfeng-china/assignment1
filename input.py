
from generate_ngram import generate_ngrams
import nltk
import sys
class LanguageModel:
    def __init__(self, n, tokens):
        self.n = n
        self.tokens = tokens
        self.n_grams = generate_ngrams(tokens, n)
        self.nprime_grams = generate_ngrams(tokens, n-1)
        self.ngram_freqDist = nltk.FreqDist(self.n_grams)
        self.nprime_freqDist = nltk.FreqDist(self.nprime_grams)

def main():
    method = sys.argv[1]
    infile = open('811_a1_train/' + 'udhr-eng.txt.tra', 'r')
    # infile = open('sentence.tra', 'r')


    corpus = infile.readlines()
    char_array = [item for sublist in corpus for item in sublist]
    # char_array = "the first the second the third the fourth the first".split(' ')
    n = 3
    model = generate_model(n, char_array)

    print "correct value is 0.0425848295499"
    print compute_mle(('e', 'a', 's'), model)

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

def generate_model(n, tokens):
    return LanguageModel(n, tokens)


def compute_mle(ngram, model):

    #TODO handle the case for unigrams

    # print ngram[0]
    # print(model.tokens.count(ngram[0]))
    # print(len(model.tokens))
    ngram_prime = ngram[0:len(ngram)-1]
    # print "ngram " + str(ngram)
    #
    # print "ngram prime " + str(ngram_prime)
    # print "ngram freq dist %f"%(model.ngram_freqDist.freq(ngram))
    # print "nprime freq dist %f"%(model.nprime_freqDist.freq(ngram_prime))
    # print len(model.tokens)
    # add this to normalize frequencies; count of nprime will be 1 off and this throws off low counts.
    normalization_factor = ((len(model.tokens)-(len(ngram)-1)) * 1.0 / (len(model.tokens)-(len(ngram_prime)-1)))
    # print "normalization factor" + str(normalization_factor)
    try:
        mle = model.ngram_freqDist.freq(ngram) / model.nprime_freqDist.freq(ngram_prime) * normalization_factor
        return mle
    except ZeroDivisionError:
        return 0.00000000000000001

#
# def laplace():
#     freqDist = nlt.FreqDist(n_grams)



if __name__ == "__main__": main()
