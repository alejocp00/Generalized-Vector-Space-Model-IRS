from src.code.search_query import search_query
from src.code.boolean_model import boolean_model
from src.code.preprocess import titles
import os
import json


def compare_models(query, query_id, data):

    results_vect0 = search_query(query)
    results_vect = []
    for i in results_vect0:
        index = titles.index(i[0])
        results_vect.append((1, index))

    results_bool = boolean_model(query)

    qrls = [[int(item) for item in sublist] for sublist in data["qrels"]]

    f_bool = confusion_matrix(results_bool, qrls, query_id)
    f_vect = confusion_matrix(results_vect, qrls, query_id)

    return f_bool, f_vect


def confusion_matrix(rec_docs, qrls, query_id):
    query_tuples = []
    relevance_metric = 1
    relevants = 0

    for q in qrls:
        if q[0] == query_id + 1:
            if q[1] < 30:
                query_tuples.append(q)
                if q[2] >= relevance_metric:
                    relevants += 1

    relevant_rec = get_relevant_rec_and_no_relevant_rec(
        query_tuples, rec_docs, relevance_metric
    )
    if len(rec_docs) == 0:
        precision = 0
    else:
        precision = len(relevant_rec) / len(rec_docs)

    if relevants == 0:
        recovered = 0
    else:
        recovered = len(relevant_rec) / relevants
    if precision + recovered == 0:
        f_pr = 0
    else:
        f_pr = 2 * precision * recovered / (precision + recovered)

    return f_pr


def get_relevant_rec_and_no_relevant_rec(query_tuples, rec_docs, relevance_metric):
    relevant_rec = set()

    for doc_tuple in rec_docs:
        doc_index = doc_tuple[1]
        for q in query_tuples:
            if q[1] == doc_index:
                if q[2] >= relevance_metric:
                    relevant_rec.add(doc_index)
    return relevant_rec


# # Obtener la ruta del directorio actual
# current_dir = os.path.dirname(os.path.abspath(__file__))

# # Construir la ruta al archivo corpus.json
# corpus_path = os.path.join(current_dir, "..", "..", "data", "corpus.json")

# with open(corpus_path, "r") as json_file:
#     data = json.load(json_file)

# i = 0
# while i < 30:
#     query = data["queries"][i]
#     compare_models(query, i)
#     i += 1
