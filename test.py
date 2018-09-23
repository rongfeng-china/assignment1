import nltk
import sys
import collections


## Add space in between words for each sentence. 
## For example: 'hello world' ---> 'h e l l o w o r l d'

def convert(sentence,n):
    words = sentence.strip().split()
    sentence_converted = '{ '*(n-1) # start symbol
    for word in words:
    	sentence_converted += ' '.join(word)
        sentence_converted += ' '
    sentence_converted += '}'   # stop symbol
    return sentence_converted

## Given a file and n 
## Generate ngrams sentence by sentence, and calculate freqDist

def generate_ngrams(corpus, n):
    fd = nltk.FreqDist('')

    for sentence in corpus: 
        sentence_converted = convert(sentence,n)
        #print sentence_converted
        ngram = nltk.ngrams(sentence_converted.split(),n)
        fd_sentence = nltk.FreqDist(ngram)
        fd += fd_sentence
    return fd
 
## Sort keys and calculate probability

def sorted_vocabulary(unique_keys):
    full_vocab = list(self.unigram_frequencies.keys())
    full_vocab.sort()
    full_vocab.append(UNK)
    return full_vocab

def main():
    #n = int(sys.argv[1])
    n = 2
    infile = open("811_a1_train/" +  '2.tra', 'r')
    corpus = infile.readlines()

    freqDist_n_minus_1 = generate_ngrams(corpus, n-1)
    freqDist = generate_ngrams(corpus, n)


    print sorted(freqDist_n_minus_1.items(), key=lambda item:item[0])

    print sorted(freqDist.items(), key=lambda item:item[0])



if __name__ == "__main__": main()
