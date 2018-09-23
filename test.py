import nltk
import sys
import collections
import math


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
## fd_1 is freqDist for (n-1)
## fd is freqdist for n
def generate_ngrams(corpus, n):
    fd = nltk.FreqDist('')
    fd_1 = nltk.FreqDist('')

    for sentence in corpus: 
        sentence_converted = convert(sentence,n)
        #print sentence_converted
        ngram = nltk.ngrams(sentence_converted.split(),n)
        ngram_1 = nltk.ngrams(sentence_converted.split(),n-1)
        fd_sentence = nltk.FreqDist(ngram)
        fd_sentence_1 = nltk.FreqDist(ngram_1)
        fd += fd_sentence
        fd_1 += fd_sentence_1
    return fd_1, fd
 
## Sort keys and calculate log probability
def calculate_prob(freqDist_n_minus_1_sorted, freqDist_sorted):
    prob = {k: float(freqDist_sorted[k]) / freqDist_n_minus_1_sorted[k[:-1]] for k in freqDist_sorted}
    return prob

## multiply probs for all test sentences
def calculate_prob_sentence(prob, sentences,n):
    result = 1.
    M = 0
    for sentence in sentences:
    	sentence_converted = convert(sentence, n)
    	ngrams = nltk.ngrams(sentence_converted.split(), n)
        ngrams = list(ngrams)
        M += len(ngrams)
        
   	for gram in ngrams:
	    result *= prob[gram]
    return result,M


## calculate perplexity for each sentence
def compute_perplexity(prob_sentence,M):
    print prob_sentence
    print M
    return prob_sentence**(-1./M)
    

def main():
    #n = int(sys.argv[1])
    n = 2
    infile = open("811_a1_train/" +  '2.tra', 'r')
    corpus = infile.readlines()

    freqDist_n_minus_1,freqDist = generate_ngrams(corpus, n)

    freqDist_n_minus_1_sorted = collections.OrderedDict(freqDist_n_minus_1)
    freqDist_sorted = collections.OrderedDict(freqDist)

    prob = calculate_prob(freqDist_n_minus_1_sorted, freqDist_sorted)
    
    devfile = open("811_a1_dev/"+ '2.dev','r')
    sentences = devfile.readlines()
    prob_sentence,M = calculate_prob_sentence(prob,sentences,n)
    
    perplexity = compute_perplexity(prob_sentence,M)
    print perplexity

if __name__ == "__main__": main()
