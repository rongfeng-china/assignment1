import unittest
from mle import *
from langid import *


import time


grams = ['the', 'first', 'the', 'second', 'the', 'third', 'the', 'fourth', 'the', 'first']

def time_mle(n, mle_method):
    from nltk.book import text8
    from nltk.book import sent8

    model = generate_model(n, text8)
    ngrams = list(generate_ngrams(sent8 * 10, model.n))
    start = time.time()
    probabilities = [mle_method(ngram, model) for ngram in ngrams]
    logProduct = reduce_to_log_prob(probabilities)
    time_elapsed = time.time() - start
    print("----- compute mle %s took %s seconds ------ " % (mle_method, time_elapsed))
    return (logProduct, time_elapsed)


class TestUnsmoothed(unittest.TestCase):

    def test_unsmooth_unigram(self):
        model = generate_model(1, grams)
        mle = compute_mle(('the', ), model)
        self.assertEquals(mle, 0.5)
        mle2 = compute_mle(('second', ), model)
        self.assertEquals(mle2, 0.1)




    def test_unsmooth_bigram(self):
        grams = ['the', 'first', 'the', 'second', 'the', 'third', 'the', 'fourth', 'the', 'first']
        model = generate_model(2, grams)
        mle = compute_mle(('the', 'first'), model)
        self.assertAlmostEqual(mle, 0.4) #some rounding errors


    def test_against_set(self):
        results = time_mle(2, compute_mle)
        self.assertAlmostEqual(results[0], -360.96979096124727)
        previous_best = 0.00065
        self.assertTrue((previous_best* 0.95)<= results[1] <= (previous_best * 1.05))





class TestLaplace(unittest.TestCase):

    def test_laplace_unigram(self):
        model = generate_model(1, grams)

        mle = compute_mle_laplace(('second', ), model)
        self.assertAlmostEqual(mle, 0.1333333)
        mle2 = compute_mle_laplace(('fifth', ), model)
        self.assertAlmostEqual(mle2, 0.0666667)

    def test_laplace_bigram(self):
        model = generate_model(2, grams)
        mle = compute_mle_laplace(('the', 'second'), model)
        self.assertAlmostEqual(mle, 0.2)

        mle2 = compute_mle_laplace(('second', 'first'), model)
        self.assertAlmostEqual(mle2, 0.1666667)

    def test_against_set(self):
        results = time_mle(2, compute_mle_laplace)
        self.assertAlmostEqual(results[0], -833.5069681)
        previous_best = 0.20
        self.assertTrue((previous_best * 0.95)<= results[1] <= (previous_best * 1.05))

class TestInterpolation(unittest.TestCase):

    def test_interpolation_unigram(self):
        model = generate_model(1, grams)

    def test_interpolation_bigram(self):
        model = generate_model(2, grams)
        mle = compute_mle_interpolation(('the', 'second'), model)
        self.assertEquals(mle, 0)

    def test_against_set(self):
        results = time_mle(2, compute_mle_interpolation)
        self.assertAlmostEqual(results[0], -360.96979096124727)
        self.assertTrue((0.00065 * 0.95)<= results[1] <= (0.00065 * 1.05))







# class TestInterpolation(unittest.TestCase):

if __name__ == '__time_mle__':
    time_mle(sys.argv[1], sys.argv[2])
if __name__ == '__main__':
    unittest.main()
    # time_mle(2, compute_mle_laplace)

    # unittest.main()
