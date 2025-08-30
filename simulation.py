from scipy.optimize import linear_sum_assignment
import random
import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
from tqdm import tqdm
import math
import itertools
import os, csv
import pickle, json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

prefix = '/home/3252395' # HPC account
import sys
sys.path.append(f'{prefix}/gtm/')

from corpus import GTMCorpus
from gtm import GTM


def generate_docs_vectorized(
    true_doc_topic_matrix, topic_word_matrix, min_words, max_words, num_docs
):
    doc_lengths = np.random.randint(min_words, max_words + 1, size=num_docs)
    max_length = np.max(doc_lengths)

    # Create a matrix of word probabilities for all documents
    word_probs = np.dot(true_doc_topic_matrix, topic_word_matrix)
    word_probs /= np.sum(word_probs, axis=1, keepdims=True)

    # Generate word indices for all documents
    random_values = np.random.random((num_docs, max_length))
    cumulative_probs = np.cumsum(word_probs, axis=1)
    word_indices = np.argmax(random_values[:, :, np.newaxis] < cumulative_probs[:, np.newaxis, :], axis=2)

    # Create a mask for valid words based on document lengths
    mask = np.arange(max_length)[np.newaxis, :] < doc_lengths[:, np.newaxis]

    # Generate words and join them into documents
    words = np.core.defchararray.add('word_', word_indices.astype(str))
    docs = np.array([' '.join(doc[mask[i]]) for i, doc in enumerate(words)])

    return docs

def generate_anchor_words(num_topics, vocab_size, anchor_words_per_topic):
    anchor_words = {}
    available_words = set(range(vocab_size))

    for topic in range(num_topics):
        if len(available_words) < anchor_words_per_topic:
            raise ValueError("Not enough unique words for anchor words")

        topic_anchors = set(random.sample(sorted(available_words), anchor_words_per_topic))
        anchor_words[topic] = topic_anchors
        available_words -= topic_anchors

    return anchor_words

def generate_documents(
    num_docs, num_topics, vocab_size, num_covs=0,
    beta=None, lambda_=None, sigma_y=0.1, sigma_topic=None,
    doc_topic_prior='dirichlet',
    min_words=50, max_words=500, anchor_words_per_topic=100,
    random_seed=None, include_y=False
):
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    topicnames = [f"Topic{i}" for i in range(num_topics)]
    docnames = [f"Doc{i}" for i in range(num_docs)]
    words = [f"word_{i}" for i in range(vocab_size)]
    cov_names = ["cov_0"] + [f"cov_{i+1}" for i in range(num_covs)]

    # Generate covariates
    if num_covs > 0:
        M_prevalence_covariates = np.column_stack((
            np.ones(num_docs),
            np.random.randint(2, size=(num_docs, num_covs))
        ))
    else:
        M_prevalence_covariates = np.ones((num_docs, 1))

    # Generate document-topic matrix
    if doc_topic_prior == 'dirichlet':
        if num_covs > 0 and lambda_ is not None:
            alpha = np.exp(np.dot(M_prevalence_covariates, lambda_))
        else:
            alpha = np.ones((num_docs, num_topics)) * 0.1
        true_doc_topic_matrix = np.array([np.random.dirichlet(a) for a in alpha])
    else:
        if num_covs > 0 and lambda_ is not None:
            mean = np.dot(M_prevalence_covariates, lambda_)
        else:
            mean = np.zeros((num_docs, num_topics))
        if sigma_topic is None:
            sigma_topic = np.eye(num_topics)
        samples = []
        for m in mean:
            sample = np.random.multivariate_normal(m, sigma_topic)
            sample = np.exp(sample) / np.exp(sample).sum()
            samples.append(sample)
        true_doc_topic_matrix = np.array(samples)

    # Generate anchor words
    anchor_words = generate_anchor_words(num_topics, vocab_size, anchor_words_per_topic)

    # Generate topic-word matrix with anchor words
    topic_word_matrix = np.zeros((num_topics, vocab_size))
    for topic in range(num_topics):
        # Identify words that can appear in this topic
        valid_words = list(anchor_words[topic].union(set(range(vocab_size)) - set().union(*anchor_words.values())))

        # Generate distribution for valid words
        dirichlet_params = np.ones(len(valid_words)) * 0.1
        topic_distribution = np.random.dirichlet(dirichlet_params)

        # Place the distribution in the correct positions in the topic-word matrix
        topic_word_matrix[topic, valid_words] = topic_distribution

    # Normalize the topic-word matrix
    topic_word_matrix /= topic_word_matrix.sum(axis=1, keepdims=True)

    # Generate documents
    documents = generate_docs_vectorized(
        true_doc_topic_matrix, topic_word_matrix,
        min_words, max_words, num_docs
    )

    # Create DataFrame
    df = pd.DataFrame({"doc": documents, "doc_clean": documents})

    # Add covariates to DataFrame
    if num_covs > 0:
        df = pd.concat([df, pd.DataFrame(M_prevalence_covariates, columns=cov_names)], axis=1)

    # Generate y if requested
    if include_y:
        if beta is None:
            beta = np.random.normal(0, 1, num_topics)
        y = np.dot(true_doc_topic_matrix, beta) + np.random.normal(0, sigma_y, num_docs)
        df['y'] = y

    # Create true distribution DataFrames
    df_doc_topic = pd.DataFrame(true_doc_topic_matrix, columns=topicnames, index=docnames)
    df_topic_word = pd.DataFrame(topic_word_matrix, index=topicnames, columns=words)
    df_true_dist_list = [df_doc_topic, df_topic_word]

    return df_true_dist_list, df