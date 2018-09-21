import nltk
import sys

def generate_ngram(n):
    infile = open("811_a1_train/" +  'udhr-eng.txt.tra', 'r')


    corpus = infile.readlines()
    char_array = [item for sublist in corpus for item in sublist]


    if n == 1:
        n_grams = nltk.word_tokenize(char_array)

    elif n == 2:
        n_grams = nltk.bigrams(char_array)
    elif n == 3:
        n_grams = nltk.trigrams(char_array)

    else:
        raise ValueError("Value must be between 1 and 3, Invalid value: " + n)
    freqDist = nltk.FreqDist(n_grams)

    # print freqDist.freq(('e', 'a'))



def main():
    n = int(sys.argv[1])
    generate_ngram(n)

if __name__ == "__main__": main()
