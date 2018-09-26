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
        filename = 'map-zero-output_{}_{}.txt'.format(n, option)
        sys.stdout = open(filename, 'w')
        identify_language(mle_method, n, training_dir, dev_dir)
        sys.stdout.close()
        sys.stdout = sys.__stdout__
        print "Finished %s"%(filename)
