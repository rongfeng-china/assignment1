import unittest
import math
from perplexity import *
class TestPerplexity(unittest.TestCase):

    def test_compute_perplexity(self):

        probabilities = [0.1] * 10
        # print probabilities
        logProduct = reduce((lambda x, y: x + y), map(lambda x: math.log(x), filter(lambda x: x > 0, probabilities)))
        # print logProduct
        perplexity = compute_perplexity(logProduct, 10)
        self.assertAlmostEqual(perplexity, 10)

if __name__ == '__main__':
    unittest.main()
