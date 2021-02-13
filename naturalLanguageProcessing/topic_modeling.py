import nltk
import gensim
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize

gen_docs = []

with open('demofile.txt') as f:
    docs = f.readlines()
    for doc in docs:
        gen_docs.append([w.lower() for w in word_tokenize(doc)])

print(gen_docs)

dictionary = gensim.corpora.Dictionary(gen_docs)
print(dictionary)
print(dictionary.token2id)

corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
# corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
print('\n', corpus)

tf_idf = gensim.models.TfidfModel(corpus)
for doc in tf_idf[corpus]:
    print([[dictionary[id], np.around(freq, decimals=2)] for id, freq in doc])

sims = gensim.similarities.Similarity('./', tf_idf[corpus],
                                      num_features=len(dictionary))

with open('demofile2.txt') as f:
    query = f.readline()
    query_doc = [w.lower() for w in word_tokenize(query)]

print('query_doc', query_doc)
query_doc_bow = dictionary.doc2bow(query_doc)
print(query_doc_bow)

query_doc_tf_idf = tf_idf[query_doc_bow]
print(query_doc_tf_idf)
# print(document_number, document_similarity)
print('Comparing Result:', sims[query_doc_tf_idf])
