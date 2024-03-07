from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.code.preprocess import preprocess_query, titles
import numpy as np
import math
import json
import os

# Obtener la ruta del directorio actual
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construir la ruta al archivo corpus.json
corpus_path = os.path.join(current_dir, "..", "..", "data", "corpus.json")

with open(corpus_path, "r") as json_file:
    data = json.load(json_file)

docs = data["original_corpus"]
filtered_docs = data["corpus"]
weight_doc_matrix = data["vector representation"]
vocabulary = data["vocabulary"]
correlation_matrix = data["correlation_matrix"]
n = len(correlation_matrix[0])


def search_query(query):
    """
    Método para buscar documentos similares a una consulta dada.

    Args:
    - query (str): La consulta de búsqueda a evaluar.

    Returns:
    - list: Lista de documentos considerados similares a la consulta.
    """
    # Preprocesar la consulta
    preprocessed_query = preprocess_query(query)

    similarity = []
    query_vector = []
    for term in vocabulary:
        query_vector.append(weight_term_in_query(term, preprocessed_query))

    # Cálculo de la similitud entre la consulta y los documentos
    for i in range(0, len(docs)):
        similarity.append(calculate_similarity(i, query_vector))

    results = []
    for i in range(0, len(similarity)):
        if similarity[i] > 0:
            results.append((titles[i], similarity[i]))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:5]


def weight_term_in_query(term, processed_query):
    """
    Calculate the weight of a term in its query based on its term frequency (TF) and inverse document frequency (IDF).

    Args:
        term (str): The term to calculate the weight for.
        processed_query (list): The preprocessed query.

    Returns:
        float: The weight of the term in the query.
    """
    # Calcular TF (Term Frequency)
    term_count = processed_query.count(term)
    tf = term_count if term_count > 0 else 0

    # Calcular IDF (Inverse Document Frequency)
    d = sum(1 for doc in filtered_docs if term in doc)
    idf = math.log(n / (d + 1))  # Añadir 1 para evitar división por cero

    # Calcular el peso del término en la consulta
    weight = tf * idf

    return weight


def calculate_similarity(doc_index, query_vector):
    """
    Calculate the similarity between a document and a query based on their vector representations,
    correlation matrix, and cosine similarity.

    Args:
        doc_index (int): Index of the document in the weight_doc_matrix to calculate the similarity with.
        query_vector (ndarray): Weighted vector representation of the query.

    Returns:
        float: The similarity score between the document and the query.
    """
    # Calcular el valor promedio de la correlación
    t_i_j = sum(sum(row) for row in correlation_matrix) / n

    # Calcular la similitud del coseno entre el vector del documento y el vector de la consulta
    return (
        cosine_similarity([weight_doc_matrix[doc_index]], [query_vector])[0][0] * t_i_j
    )


# print ("Ingrese una consulta de búsqueda:")
# query = input()

# # Buscar documentos similares a la consulta
# results = search_query(query)

# # Imprimir los resultados
# print ("Resultados:")
# for i, result in enumerate(results):
#     print (f"Resultado {i+1}: {result}")
