import nltk
import sys

def generate_ngrams(tokens, n):

    n_grams = nltk.ngrams(tokens, n)
    return n_grams


def main():
    n = int(sys.argv[1])
    infile = open("811_a1_train/" +  'udhr-eng.txt.tra', 'r')


    corpus = infile.readlines()
    corpus.join()
    char_array = [item for sublist in corpus for item in "\n" + sublist] # add start character

    n_grams = generate_ngrams(char_array, n)
    freqDist = nltk.FreqDist(n_grams)

    print freqDist.freq(('e', 'a'))
    print freqDist.freq(('a', 'e'))


if __name__ == "__main__": main()
