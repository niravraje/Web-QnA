#!/usr/bin/env python
# coding: utf-8

# ## Here we first load the glove vectors as a dictionary - `embeddings_index`
# `embeddings_index['banana']` would give some 100 length vector for the word `'banana'`
# 
# The object `GLOVE_DIR` points to the text file which containes the vectors, but it could also be downloaded form http://nlp.stanford.edu/data/glove.6B.zip and saved on disk

import os
import numpy as np
GLOVE_DIR ='Path to directory containing glove txt file'
print('Indexing word vectors.')
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'),encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

print(embeddings_index['banana'])


# ## Let's find the top 7 words that are closest to 'compute'

u = embeddings_index['compute']
norm_u = np.linalg.norm(u)
similarity = []

for word in embeddings_index.keys():
    v = embeddings_index[word]
    cosine = np.dot(u, v)/norm_u/np.linalg.norm(v)
    similarity.append((word, cosine))
print(len(similarity))

sorted(similarity, key=lambda x: x[1], reverse=True)[:10]


# ## Now let's do vector algebra.
# 
# ### First we subtract the vector for `france` from `paris`. This could be imagined as a vector pointing from country to its capital. Then we add the vector of `nepal`. Let's see if it does point to the country's capital

output = embeddings_index['paris'] - embeddings_index['france'] + embeddings_index['nepal']
norm_out = np.linalg.norm(output)

similarity = []
for word in embeddings_index.keys():
    v = embeddings_index[word]
    cosine = np.dot(output, v)/norm_out/np.linalg.norm(v)
    similarity.append((word, cosine))


print(len(similarity))

sorted(similarity, key=lambda x: x[1], reverse=True)[:7]





