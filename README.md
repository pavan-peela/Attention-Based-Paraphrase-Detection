# Paraphrase Detection using Attention Mechanism

## Base Model
We use the bilateral multi-perspective matching (BiMPM) model by Zhiguo Wang, Wael Hamza, and Radu Florian.
Given two sentences P and Q, the model first encodes them with a BiLSTM encoder. Next, the two encoded
sentences are matched in two directions P against Q and Q against P . In each matching direction, each time
step of one sentence is matched against all time steps of the other sentence from multiple perspectives.
Then, another BiLSTM layer is utilized to aggregate the matching results into a fixed-length
matching vector. Finally, based on the matching vector, a decision is made through a fully connected layer.

## Proposed Attention Based Model
We add a linear attention layer after the Aggregation layer to extract meaningful phrases from the sentences,
and increase the weightage of that part of the sentence for matching.


## Datasets used
### MSRP Dataset
The dataset consists of 5,801 sentence pairs.  The shortest sentence has 7 words and the longest 36.
3,900 are labeled as positive for being a paraphrase. Out of the total, 4,076 are training
pairs (67.5% are paraphrases) and 1,725 test pairs (66.5% are paraphrases).


## Download links for pretrained word embeddings :
The following embeddings needs to be download to get started.

1. Glove : https://nlp.stanford.edu/projects/glove/
  - We specifically used Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors)
  
2. Word2vec : https://code.google.com/archive/p/word2vec/


## How to run :
1. Download Glove embeddings from above links, and place the .txt file in the root folder.
2. Running any of the files for the first time would extract the glove embeddings and put them in cache/.
3. For subsequent runs, the model will just load the embeddings from the cache.
4. Two models (GRU and LSTM) each with and without an Attention Layer have been implemented.


                                        -------- X  Pavan Kumar Peela  X --------
