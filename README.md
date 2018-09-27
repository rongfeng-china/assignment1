# CMPUT 650 Assignment 1

Daniel Chui dchui1@ualberta.ca

Rong Feng rfeng2@ualberta.ca

## Usage

Tested on python 2.7.11

Run using `python langid.py <mode> <training_dir> <test_dir>`
where `<mode>` is `--unsmoothed, --laplace, or --interpolation`
and `<training_dir>` is the path to the training directory
and `<test_dir>` is the path to the test directory



For example, to identify the language using an unsmoothed language model
with the training files in `811_a1_train` and the test files located at `811_a1_test`
you would run
```python langid.py --unsmoothed 811_a1_train/ 811_a1_test/```

The expected output would be something like this:
```udhr-als.txt.dev udhr-als.txt.tra 20.214536 1```
where the first column lists the name of the test file, the 2nd column lists the
training file with the lowest perplexity for the test file, the 3rd column lists
the perplexity, and the 4th column shows the value of *n* for the n-grams used
to train the language model.

NOTE: For single files, you must create the appropriate directories and place the file within the associated directory.

## Other Files

`generate_ngram.py`, `mle.py`, `perplexity.py`, and `langid.py` are required to run `langid`. The other files included are unit tests and utilities used during the parameter tuning portion of the assignment
