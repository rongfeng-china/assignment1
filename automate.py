from langid import *
from mle import *
import sys
options = ["--unsmoothed", "--laplace", "--interpolation"]

training_dir = '811_a1_train/'
# dev_dir = '811_a1_dev/'
dev_dir = '811_a1_dev/'
for option in options:
    for n in range(1, 6):
        mle_method = get_method(option)
        sys.stdout = open('output_{}_{}.txt'.format(n, option), 'w')
        identify_language(mle_method, n, training_dir, dev_dir)
        sys.stdout.close()
