import string
import random
import numpy as np
import scipy as sp
import urllib.request
import nltk
import re
import heapq
from bs4 import BeautifulSoup
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

def get_article_text(url):
    # Scrape article using bs4 to extract all paragraphs from the online article.
    raw_html = urllib.request.urlopen(url)
    raw_html = raw_html.read()

    article_html = BeautifulSoup(raw_html, 'lxml')
    article_paragraphs = article_html.find_all('p')

    # Creating a document 'article_text' containing all the sentences in the article.
    article_text = ''
    for para in article_paragraphs:
        article_text += para.text
    return article_text

def remove_stopwords(sentence):
    filtered_sentence = []
    stop_words = nltk.corpus.stopwords.words('english')
    word_tokens = nltk.word_tokenize(sentence)
    for token in word_tokens:
        if token not in stop_words:
            filtered_sentence.append(token)
    filtered_sentence = ' '.join(filtered_sentence)
    return filtered_sentence

def clean_sentence(sentence):
    sentence = sentence.lower()
    sentence = remove_stopwords(sentence)
    sentence = re.sub(r'\W', ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence

def clean_article_text(article_text):
    # Creating a corpus containing all the sentence tokens in the document.
    corpus = nltk.sent_tokenize(article_text)
    # Convert to lowercase, remove non-word characters (punctuations, etc.) and strip whitespaces
    for i in range(len(corpus)):
        corpus[i] = clean_sentence(corpus[i])
    return corpus

def create_word_freq_dictionary(corpus):
    # Create dictionary with word frequency
    word_freq = defaultdict(int)
    for sentence in corpus:
        word_tokens = nltk.word_tokenize(sentence)
        for token in word_tokens:
            word_freq[token] += 1
    return word_freq

def generate_sent_vec(sentence, most_freq_tokens):
    word_tokens = nltk.word_tokenize(sentence)
    sent_vec = []
    for token in most_freq_tokens:
        if token in word_tokens:
            sent_vec.append(1)
        else:
            sent_vec.append(0)
    return sent_vec

def get_sentence_vectors(corpus, most_freq_tokens):
    # Generate sentence vectors of 1's and 0's. Feature set is the most_freq_tokens list.
    sentence_vectors = []
    for sentence in corpus:
        sent_vec = generate_sent_vec(sentence, most_freq_tokens)
        sentence_vectors.append(sent_vec)
        
    sentence_vectors = np.asarray(sentence_vectors)
    return sentence_vectors

def get_answer(url, question):

    article_text = get_article_text(url)
    #print("Article Text: \n", article_text)
    initial_corpus = nltk.sent_tokenize(article_text)
    corpus = clean_article_text(article_text)

    word_freq = create_word_freq_dictionary(corpus)

    # Get the most frequent tokens from the dictionary
    most_freq_tokens = heapq.nlargest(200, word_freq, key=word_freq.get)

    sentence_vectors = get_sentence_vectors(corpus, most_freq_tokens)

    cleaned_question = clean_sentence(question)
    question_vector = generate_sent_vec(cleaned_question, most_freq_tokens)

    similarity_scores = []
    sent_vec_index = 0
    for sent_vec in sentence_vectors:
        similarity = 1 - sp.spatial.distance.cosine(question_vector, sent_vec)
        similarity_scores.append((sent_vec_index, similarity))
        sent_vec_index += 1
    similarity_scores.sort(key = lambda x: x[1], reverse=True)
    answer_index = similarity_scores[0][0]

    return initial_corpus[answer_index]