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

    query_vector = [tf_idf(term, preprocessed_query) for term in vocabulary]

    # Cálculo de la similitud entre la consulta y los documentos
    similarity = [calculate_similarity(i, query_vector) for i in range(len(docs))]

    # Enlistar los titulos y similitud de los resultados
    results = [(titles[i], sim) for i, sim in enumerate(similarity) if sim >= 1]
    results.sort(reverse=True)
    return results


def tf_idf(term, processed_query):

    term_count = processed_query.count(term)
    tf = term_count if term_count > 0 else 0
    d = sum(1 for doc in filtered_docs if term in doc)
    idf = np.log(n / (d + 1))  # Añadir 1 para evitar división por cero
    return tf * idf

    # Calcular IDF (Inverse Document Frequency)
    d = sum(1 for doc in filtered_docs if term in doc)
    idf = math.log(n / (d + 1))  # Añadir 1 para evitar división por cero

    # Calcular el peso del término en la consulta
    weight = tf * idf

    return weight


def calculate_similarity(doc_index, query_vector):

    # Calcular el valor promedio de la correlación
    t_i_j = sum(sum(row) for row in correlation_matrix) / n

    # Calcular la similitud del coseno entre el vector del documento y el vector de la consulta
    return (
        cosine_similarity([weight_doc_matrix[doc_index]], [query_vector])[0][0] * t_i_j
    )
