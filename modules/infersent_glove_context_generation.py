import time
import math
import numpy as np
import urllib.request
import nltk
nltk.download('punkt')
import re
import torch
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
from models_infersent import InferSent

def get_sentence_tokens(url):
    # Scrape article using bs4 to extract all paragraphs from the online article.
    raw_html = urllib.request.urlopen(url)
    raw_html = raw_html.read()

    article_html = BeautifulSoup(raw_html, 'lxml')
    article_paragraphs = article_html.find_all('p')
    
    # Creating a document 'article_text' containing all the sentences in the article.
    article_text = ''
    for para in article_paragraphs:
        article_text += para.text
        
    # Tokenize article text into sentences.
    article_sentences = nltk.sent_tokenize(article_text)
    
    # Clean the article sentence to remove extra whitespaces and reference numbers (such as "[23]")
    for i in range(len(article_sentences)):
        article_sentences[i] = re.sub(r'\[\d+\]', '', article_sentences[i])
        article_sentences[i] = re.sub(r'\[\w\]', '', article_sentences[i])
        article_sentences[i] = re.sub(r'\s+', ' ', article_sentences[i]).strip()
    
    return article_sentences

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def euclidean_dist(u, v):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(u, v)]))

def get_most_similar_sentences(question_vector, embeddings, sent_count):
    """Returns the most similar sentences to the question vector.
    Similarity Coefficient used: Cosine Index
    Sentence count refers to number of most similar sentences to be returned.
    """
    most_sim_sentences = []
    for sent_index, sent_vector in enumerate(embeddings):
        most_sim_sentences.append((sent_index, cosine(question_vector, sent_vector))) # appending a tuple
    most_sim_sentences.sort(key = lambda x: x[1], reverse = True)
    
    assert sent_count <= len(embeddings), 'Enter sent_count value less than or equal to {0}'.format(len(embeddings))
    return most_sim_sentences[:sent_count]

def prepare_model():
    # Load the InferSent model
    model_version = 1
    MODEL_PATH = "C:/Users/niraje/Documents/MLG/InferSent/encoder/infersent%s.pkl" % model_version
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
    model = InferSent(params_model)
    model.load_state_dict(torch.load(MODEL_PATH))
    
    # Set the GloVe directory path:
    W2V_PATH = 'C:/Users/niraje/Documents/MLG/word_vectors/glove/glove.6B.300d.txt'
    model.set_w2v_path(W2V_PATH)
    
    # Load embeddings of K most frequent words
    model.build_vocab_k_words(K=100000)
    
    return model

def generate_context(url, question):
    
    # Get the sentence tokens for the entire article text.
    article_sentences = get_sentence_tokens(url)
    
    # Get the prepared model using GloVe/InferSent embeddings as its vocabulary.
    model = prepare_model()
    
    # Encode sentences
    embeddings = model.encode(article_sentences, bsize=128, tokenize=False, verbose=True)
    
    # Encode the question
    question = [question]
    question_vector = model.encode(question, bsize=128, tokenize=False, verbose=True)[0]
    
    # Get most similar "N" sentence tokens i.e. sent_count
    most_sim_sentences = get_most_similar_sentences(question_vector, embeddings, sent_count = 30)
    
    # Build context paragraph.
    # Choose max_token_count such that total token count (question and context) is < 512.\
    context_list = []
    context_token_count = 0
    max_token_count = 400

    for sent_index, similarity_score in most_sim_sentences:
        sent_token_count = len(nltk.word_tokenize(article_sentences[sent_index]))
        if context_token_count + sent_token_count < max_token_count:
            context_list.append(article_sentences[sent_index])
            context_token_count += sent_token_count

    context_para = ' '.join(context_list)
    
    return context_para