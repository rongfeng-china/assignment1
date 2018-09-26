import unittest
from mle import *

import time


grams = ['the', 'first', 'the', 'second', 'the', 'third', 'the', 'fourth', 'the', 'first']



class TestUnsmoothed(unittest.TestCase):

    def test_unsmooth_unigram(self):
        model = generate_model(1, grams)
        mle = compute_mle_manually(('the', ), model)
        self.assertEquals(mle, 0.5)

        mle2 = compute_mle_manually(('second', ), model)
        self.assertEquals(mle2, 0.1)


    def test_unsmooth_bigram(self):
        grams = ['the', 'first', 'the', 'second', 'the', 'third', 'the', 'fourth', 'the', 'first']
        model = generate_model(2, grams)
        mle = compute_mle(('the', 'first'), model)
        self.assertAlmostEqual(mle, 0.4)
        manmle = compute_mle_manually(('the', 'first'), model)
        self.assertEquals(manmle, 0.4)

    def compare_mle(self):

        from nltk.book import text1
        model = generate_model(2, text1)

        ngrams = list(generate_ngrams(text1, model.n))
        print("length of tokens %d"%(len(ngrams)))

        start = time.time()
        mle_method = compute_mle
        probabilities = [mle_method(ngram, model) for ngram in ngrams]

        print("----- compute mle %s seconds ------ " % (time.time() - start))
        # start = time.time()
        # mle_method = compute_mle_manually
        # probabilities = [mle_method(ngram, model) for ngram in ngrams]
        #
        # print("----- compute mle manually %s seconds ------ " % (time.time() - start))
def time_mle(n, mle_method):
    from nltk.book import text1
    model = generate_model(n, text1)
    ngrams = list(generate_ngrams(text1, model.n))
    start = time.time()
    probabilities = [mle_method(ngram, model) for ngram in ngrams]
    print("----- compute mle %s took %s seconds ------ " % (mle_method, (time.time() - start)))



class TestLaplace(unittest.TestCase):

    def test_laplace_unigram(self):
        model = generate_model(1, grams)

    def test_laplace_bigram(self):
        model = generate_model(2, grams)
        mle = compute_mle_laplace(('the', 'second'), model)
        self.assertAlmostEqual(mle, 0.2)

        # mle2 = compute_mle(('second', 'first'), model)
        # self.assertAlmostEqual(mle2, 0.0667)

class TestInterpolation(unittest.TestCase):
    





# class TestInterpolation(unittest.TestCase):

if __name__ == '__time_mle__':
    time_mle(sys.argv[1], sys.argv[2])
if __name__ == '__main__':
    time_mle(2, compute_mle_laplace)

    # unittest.main()
