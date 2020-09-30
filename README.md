This fork of [JoeyNMT](https://github.com/joeynmt/joeynmt) is a term project by Jennifer Mell for the course "Introduction to Neural Networks and Sequence-to-Sequence Learning" at Heidelberg University.
It implements an unsupervised neural machine translation architecture as described in [Unsupervised Neural Machine Translation, Artetxe et al. (2017)](http://arxiv.org/abs/1710.11041).

An example configuration for an unsupervised NMT model can be found in `configs/unsupervised_small.yaml`.

Four scripts were added to the scripts folder:
* `preprocess.py` normalises and tokenises a corpus using the sacremoses language-specific normaliser and tokeniser
* `filter.py` filters sentences containing less than the minimum or more than the maximum number of tokens specified.
* `noise.py` creates a noised parallel corpus for the denoising task
* `plot_vocab_coverage.py` plots the vocabulary coverage of a model

The training data used for the experiments was sampled from [Leipzig Corpora Collection](https://wortschatz.uni-leipzig.de/de/download) news corpora for German and English.
The bilingual development and test data was sampled from [WMT'14 News Commentary](http://www.statmt.org/wmt14/translation-task.html) data.
