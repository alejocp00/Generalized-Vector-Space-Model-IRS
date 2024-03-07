import ir_datasets
import nltk
import gensim
import pandas as pd
from scipy.stats import chi2_contingency
import numpy as np
from math import log
import os
import json

# Descargar el conjunto de palabras vacías si aún no está disponible
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
titles = []

#Cargar el dataset
def load_dataset():
    """
    Load the dataset from the "ir_datasets" library.

    Returns:

    """
    dataset = ir_datasets.load("cranfield")

    documents = []
    queries = []
    qrels = []

    for i, doc in enumerate(dataset.docs_iter()):
        if i == 30:
            break
        documents.append(doc.text)
        titles.append(doc.title)
    
    for i, query in enumerate(dataset.queries_iter()):
        if i == 30:
            break
        queries.append(query.text)

    for i, qrel in enumerate(dataset.qrels_iter()):
        qrels.append(qrel)
    
    return documents, queries, qrels

def preprocess_data(documents, queries, qrels):
    """
    Preprocess a list of documents and saving the result data to a JSON file.

    Args:
        documents (list): A list of documents to be preprocessed.
        queries 
        qrels

    Returns:
        None
    """
    
    tokenized_docs = tokenization(documents) 
    cleaned_docs = remove_noise(tokenized_docs)
    no_stop_words_docs = remove_stopwords(cleaned_docs)
    reduced_docs = morphological_reduction(no_stop_words_docs)
    filtered_docs, dictionary = filter_tokens_by_occurrence(reduced_docs)
    vocabulary = build_vocabulary(dictionary)
    vector_rep = vector_representation(filtered_docs, vocabulary)
    vectorial_docs = docs_vectorial_rep(vocabulary, filtered_docs)
    correlation_matrix = get_correlation_matrix(vectorial_docs)

    # Obtener la ruta del directorio actual
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construir la ruta al archivo corpus.json
    corpus_path = os.path.join(current_dir,'..', '..', 'data', 'corpus.json')

    #Guardar documento preprocesado en el json
    data = {"original_corpus": documents,
            "queries": queries,
            "qrels": qrels,
            "corpus": filtered_docs,
            "vector representation": vector_rep,
            "vocabulary" : vocabulary,
            "correlation_matrix": correlation_matrix}
    
    with open(corpus_path, "w") as json_file:
        json.dump(data, json_file)

def preprocess_query(query):
    """
    Args:
        query (str): The input query to be preprocessed.

    Returns:
        list: A list of tokens representing the vocabulary after preprocessing the query.
    """
    tokenized_query = tokenization([query])
    cleaned_query = remove_noise(tokenized_query)
    no_stop_words_query = remove_stopwords(cleaned_query)
    reduced_query = morphological_reduction(no_stop_words_query)
    filtered_query, dictionary = filter_tokens_by_occurrence(reduced_query)
    vocabulary = build_vocabulary(dictionary)
    return vocabulary


# Funciones para preprocesar un conjunto de documentos

def tokenization(documents):
    """

    Args:
        documents (list): A list of documents to tokenize.

    Returns:
        list: A list of lists of tokens, where each inner list represents the tokens of a document.
    """
    return [nltk.word_tokenize(doc) for doc in documents]


def remove_noise(tokenized_docs):
    """
    Remove non-alphabetic tokens from a list of tokenized documents.

    Args:
        tokenized_docs (list): A list of tokenized documents.

    Returns:
        list: A list of tokenized documents with non-alphabetic tokens removed.
    """
    return [[token for token in list if token.isalpha()] for list in tokenized_docs]


def remove_stopwords(tokenized_docs):
    """
    Remove stopwords from a list of tokenized documents.

    Args:
        tokenized_docs (list): A list of tokenized documents.

    Returns:
        list: A list of tokenized documents with stopwords removed.
    """
    stopwords = set(nltk.corpus.stopwords.words('english'))
    return [[token for token in list if token not in stopwords] for list in tokenized_docs]

def morphological_reduction(tokenized_docs, use_lemmatization=True):
    """
    Perform morphological reduction on a list of tokenized documents.

    Args:
        tokenized_docs (list): A list of tokenized documents.
        use_lemmatization (bool, optional): Flag indicating whether to use lemmatization (default: True).

    Returns:
        list: A list of tokenized documents with morphological reduction applied.
    """
    if use_lemmatization:
        lemmatizer = nltk.WordNetLemmatizer().lemmatize
        return [[lemmatizer(token) for token in list] for list in tokenized_docs]
    else:
        stemmer = nltk.PorterStemmer().stem
        return [[stemmer(token) for token in list] for list in tokenized_docs]


def filter_tokens_by_occurrence(tokenized_docs, no_below=1, no_above=20):
    """
    Filter tokens in a list of tokenized documents based on their occurrence frequency.

    Args:
        tokenized_docs (list): A list of tokenized documents.
        no_below (int, optional): Minimum number of occurrences for a token to be included.
        no_above (float, optional): Maximum proportion of documents a token can appear in.

    Returns:
        tuple: A tuple containing the filtered tokens and the dictionary used for filtering.
            - filtered_tokens (list): A list of tokenized documents with filtered tokens.
            - dictionary (gensim.corpora.Dictionary): The dictionary used for filtering.
    """
    dictionary = {}
    dictionary = gensim.corpora.Dictionary(tokenized_docs)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)

    filtered_words = [word for _, word in dictionary.iteritems()]
    filtered_tokens = [
        [word for word in doc if word in filtered_words]
        for doc in tokenized_docs
    ]

    return filtered_tokens, dictionary


def build_vocabulary(dictionary):
    """
    Build a vocabulary list from a dictionary object.

    Args:
        dictionary (gensim.corpora.Dictionary): A dictionary object.

    Returns:
        list: A list of tokens representing the vocabulary.
    """
    vocabulary = list(dictionary.token2id.keys())
    return vocabulary


def vector_representation(filtered_docs, vocabulary):
    """
    Generate vector representations of filtered documents using term frequency-inverse document frequency (TF-IDF).

    Args:
        filtered_docs (list): A list of tokenized documents with filtered tokens.
        vocabulary (list): A list of tokens representing the vocabulary.

    Returns:
        list: A list of vector representations of the filtered documents.
    """
    vector_repr=[]
    term_frequency = []
    
    for doc in filtered_docs:
        doc_i=[]
        for voc in vocabulary:
            term_frequency.append(doc.count(voc))

        for word in vocabulary:
            f=doc.count(word)
            tf = f / max(term_frequency)
            idf=log(len(filtered_docs)/(1+sum([1 for doc in filtered_docs if word in doc])))
            doc_i.append(tf*idf)
        vector_repr.append(doc_i)

    return vector_repr


def pos_tagger(tokenized_docs):
    """
    Perform part-of-speech tagging on a list of tokenized documents.

    Args:
        tokenized_docs (list): A list of tokenized documents.

    Returns:
        list: A list of lists of tuples, where each inner list represents the part-of-speech tagged tokens of a document.
              Each tuple contains the token text and its corresponding part-of-speech tag.
    """
    return [
        [(token.text, token.tag_) for token in doc]
        for doc in tokenized_docs
    ]


def docs_vectorial_rep(vocabulary, filtered_tokens):
    """
    Generate vectorial representations of documents based on the presence or absence of vocabulary tokens.

    Args:
        vocabulary (list): A list of tokens representing the vocabulary.
        filtered_tokens (list): A list of tokenized documents with filtered tokens.

    Returns:
        list: A list of vectorial representations of the documents.
    """
    vectorial_docs = []
    for doc in filtered_tokens:
        doc_rep = []
        for voc in vocabulary:
            if voc in doc:
                doc_rep.append(1)
            else: 
                doc_rep.append(0)
        vectorial_docs.append(doc_rep)
    return vectorial_docs


def get_correlation_between_terms(vectorial_docs, term_i, term_j):
    """
    Calculate the correlation between two terms based on their co-occurrence in vectorial documents.

    Args:
        vectorial_docs (list): A list of vectorial representations of documents.
        term_i (int): Index of the first term in the vectorial representation.
        term_j (int): Index of the second term in the vectorial representation.

    Returns:
        float: The correlation between the two terms, measured using the phi coefficient.
    """
    df = pd.DataFrame(vectorial_docs)
    filtered_df = df[[term_i, term_j]]
    contingency_table = pd.crosstab(filtered_df[term_i].values.tolist(), filtered_df[term_j].values.tolist())
    observed = contingency_table.values
    chi2, _, _, expected = chi2_contingency(observed)
    phi = np.sqrt(chi2 / np.sum(observed))
    return phi


def get_correlation_matrix(vectorial_docs):
    """
    Calculate the correlation matrix between terms based on their co-occurrence in vectorial documents.

    Args:
        vectorial_docs (list): A list of vectorial representations of documents.

    Returns:
        list: A correlation matrix where each element represents the correlation between two terms.
              The matrix is symmetric and has a diagonal of 1.
    """
    correlation_matrix = []
    for i in range(len(vectorial_docs[0])):
        doc_correlation = []
        for j in range(len(vectorial_docs[0])):
            if i<=j:
                if i == j:
                    doc_correlation.append(1)
                else:
                    doc_correlation.append(get_correlation_between_terms(vectorial_docs, i, j))
            else:
                doc_correlation.append(correlation_matrix[j][i])
        correlation_matrix.append(doc_correlation)
        print (i)
    return correlation_matrix

load_dataset()
#preprocess_data(docs, queries, qrels)