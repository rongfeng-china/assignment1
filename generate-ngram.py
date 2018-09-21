import nltk
import sys

def generate_ngrams(tokens, n):


    if n == 1:
        n_grams = nltk.word_tokenize(tokens)

    elif n == 2:
        n_grams = nltk.bigrams(tokens)
    elif n == 3:
        n_grams = nltk.trigrams(tokens)

    else:
        raise ValueError("Value must be between 1 and 3, Invalid value: " + n)

    return n_grams




def main():
    n = int(sys.argv[1])
    infile = open("811_a1_train/" +  'udhr-eng.txt.tra', 'r')


    corpus = infile.readlines()
    char_array = [item for sublist in corpus for item in sublist]

    n_grams = generate_ngrams(char_array, n)
    freqDist = nltk.FreqDist(n_grams)

    print freqDist.freq(('e', 'a'))

if __name__ == "__main__": main()
