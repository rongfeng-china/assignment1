
from generate_ngram import generate_ngrams
import nltk
import sys

def main():
    method = sys.argv[1]
    infile = open("811_a1_train/" +  'udhr-eng.txt.tra', 'r')


    corpus = infile.readlines()
    char_array = [item for sublist in corpus for item in sublist]
    n = 2
    n_grams = generate_ngrams(char_array, n)


    if method == "--unsmoothed":
        ngram_list = list(n_grams)
        # vocabCount = len(set(char_array))
        ngram_count = ngram_list.count(ngram)
        ngram_count / len(ngram_list)

    elif method == "--laplace":
        freqDist = nltk.FreqDist(n_grams)
        freqDist = nltk.LaplaceProbDist(freqDist)
    # elif method == "--interpolation":
    #     # freqDist = interpolation()
    else:
        raise ValueError("Invalid smoothign method: " + method)

    print freqDist.freq(('e', 'a'))
    print freqDist.prob(('a', 'e'))
#
# def laplace():
#     freqDist = nlt.FreqDist(n_grams)



if __name__ == "__main__": main()
