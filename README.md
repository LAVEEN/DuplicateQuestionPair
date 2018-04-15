
## Setup
The project is based on Deep Learning Neural Network.It uses keras and tensorflow. It has been tested on Ubuntu 16.04 LTS.


### Requirements

  * Python 3.3+ or Python 2.7
  * macOS or Linux or Windows

### Dependencies :

1.  Python
2.  Keras
3.  Pandas
4.  Matplotlib
5.  Hdf5
6.  H5py
7.  Nltk.corpus
8.  re
9.  Pickle
10. Numpy
11. Glove Word2Vec


#### Instructions:

1. Place glove folder in extracted form in the same directory of codes.

2. Download quora dataset from “http://qim.ec.quoracdn.net/quora_duplicate_questions.tsv”.
Place it in the same directory of codes in extracted form. Name of this dataset should be different from “train.csv” and “test.csv” 


## Run:
1. python make_test_train_data_from_given_quora_dataset <dataset_name.csv>


2. For LSTM with word embedding:

	For training :  python siamese_lstm_word.py train
	
	For testing  :  python siamese_lstm_word.py test


  For LSTM with char embedding:
  
	python char_embedding.py
	
	For training :  python siamese_lstm_char.py train
	
	For testing  :  python siamese_lstm_char.py test


  For BiLSTM with word embedding:
  
  	For training :  python siamese_BiLSTM_word.py train
	
	For testing  :  python siamese_BiLSTM_word.py test





Documentation
============

The documentation of the latest released version of Duplicate Question Pair Detection is available [here](https://docs.google.com/document/d/10rPiUkijm7ukeQcE-w_2ldIxhBeIxyDVHNKg8uabrFQ/edit?usp=sharing). 
