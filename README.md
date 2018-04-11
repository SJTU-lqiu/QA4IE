# QA4IE
This is the implementation of the paper [QA4IE: A Question Answering based Framework for Information Extraction](https://arxiv.org/abs/1804.03396).

This reporsitory follows the implementation of [BiDAF](https://github.com/allenai/bi-att-flow).

Please contact Lin Qiu(lqiu@apex.sjtu.edu.cn) for questions and suggestions.

## Requirements

Python 3.6

Python packages

tensorflow==1.0.0

Jinja2==2.9.5

MarkupSafe==0.23

numpy==1.12.0

protobuf==3.2.0

six==1.10.0

tensorflow-gpu==1.0.0

tqdm==4.11.2

nltk==3.2.1

## Datasets

We provide the [QA4IE Benchmark](https://drive.google.com/file/d/12dZQqDTNY0pSpJKkLG6JWgGcEqQu2PkZ/view) in our paper. It is a document level IE benchmark in a readable text format very similar to [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/). You shold download and unzip this file to ``$Home/data``.

To run our code, you need to download [GloVe](http://nlp.stanford.edu/data/glove.6B.zip) for pre-trained word embedding and NLTK for tokenizer. You can run ``download.sh`` to download these two datasets to ``$Home/data``.

## Preprocessing

Run the preprocessing code at ``squad/prepro_span.py`` for span datasets and ``squad/prepro_seq.py`` for seq datasets with:

	python -m squad.prepro_span

and
	
	python -m squad.prepro_seq

The data file after preprocessing will be saved in ``$PWD/data/qa4ie``. You need to make sure the argument of input file path are correct in the code (line 26). The default setting is for small sized datasets with document length < 400. If you want to try datasets with longer documents, you should modify the source_dir with ``--source_dir``.

## Train

To train our model in default settings, run:

	python -m basic.cli --mode train --noload --len_opt --cluster --run_id default

The default setting is to train a model with a span dataset. Additional configurations can be found at ``basic/cli.py``.

## Test

To test, run:

	python -m basic.cli --len_opt --cluster --run_id default

This command loads the most recently saved model during training and begins testing on the test data. You can find the inference results of test data in the output directory ``$PWD/out/basic/default/answer``.

## Evaluate

To evaluate, run:

	python squad/evaluate-v1.1.py <file dir of groundtruths> <file dir of inference results>

The file directories here in default settings are:

	python squad/evaluate-v1.1.py $HOME/data/span/0-400/test.span.json $PWD/out/basic/default/answer/test-100000.json

The evaluation results can be found in ``$PWD/test_result``.