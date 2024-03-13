from src.code.preprocess import tokenization
from src.code.search_query import filtered_docs
import nltk
from sympy import sympify, to_dnf, Not, And, Or


def boolean_model(query):
    preprocessed_query = tokenization([query])

    query = convert_to_logic(query)
    query = query_to_dnf(query)

    # Realizar la consulta
    matching_documents = get_matching_docs(query)

    results = []

    # Convertir matching_documents a lista
    matching_documents = list(matching_documents)

    for i in range(0, len(matching_documents)):
        results.append((1, matching_documents[i]))

    return results


def convert_to_logic(query):

    new_doc = []

    for token in query:
        if token.isalpha():
            new_doc.append(token)

    logical_query = []

    for i in range(0, len(new_doc)):
        logical_query.append(new_doc[i])
        if i < len(new_doc) - 1:
            logical_query.append("or")

    doc = logical_query
    # Lista para almacenar los términos clave
    terms = []

    # Recorrer los tokens de la consulta
    for token in doc:
        # Verificar si el token es un sustantivo o un adjetivo
        if token == "and":
            terms.append("AND")
        elif token == "or":
            terms.append("OR")
        elif token == "not":
            terms.append("NOT")
        else:
            terms.append(token)

    # Construir la expresión lógica
    logic_expr = " ".join(terms)

    return logic_expr


def query_to_dnf(query):

    processed_query = ""

    # Reemplazar los operadores lógicos por sus equivalentes en sympy
    processed_query = query
    processed_query = (
        processed_query.replace("AND", "&").replace("OR", "|").replace("NOT", "~")
    )

    # Convertir a expresión sympy y aplicar to_dnf
    query_expr = sympify(processed_query, evaluate=False)
    query_dnf = to_dnf(query_expr, simplify=True, force=True)

    return query_dnf


def get_matching_docs(query_dnf):

    setDnf = set()

    if not query_dnf.args:
        term = term_in_docs(query_dnf)
        if term:
            setDnf.add(tuple(term))

    elif len(query_dnf.args) == 1:
        term = not_term_in_docs(query_dnf.args[0])
        if term:
            setDnf.add(tuple(term))

    else:
        for term in query_dnf.args:

            if isinstance(term, Or):
                terms = []
                for subterm in term.args:
                    if isinstance(subterm, Not):
                        terms.append(not_term_in_docs(subterm.args[0], corpus))
                    else:
                        terms.append(term_in_docs(subterm, corpus))

                setTemp = set()
                for i in terms:
                    setTemp = setTemp | set(tuple(i))

                setDnf.add(setTemp)

            elif isinstance(term, And):

                terms = []
                for subterm in term.args:
                    if isinstance(subterm, Not):
                        terms.append(not_term_in_docs(subterm.args[0]))
                    else:
                        terms.append(term_in_docs(subterm))

                setTemp = set(tuple(terms[0]))
                for i in terms:
                    setTemp = setTemp & set(tuple(i))

                setDnf.add(tuple(setTemp))

            elif isinstance(term, Not):
                term = not_term_in_docs(term.args[0])
                setDnf.add(tuple(term))

            else:
                term = term_in_docs(term)
                setDnf.add(tuple(term))

    if setDnf:
        firstSet = set(setDnf.pop())
        for i in setDnf:
            firstSet = firstSet | set(i)
        setDnf = firstSet

    return setDnf


def not_term_in_docs(term):
    docs = []
    for i in range(len(filtered_docs)):
        if str(term) not in filtered_docs[i]:
            docs.append(i)
    return docs


def term_in_docs(term):
    docs = []
    for i in range(len(filtered_docs)):
        if str(term) in filtered_docs[i]:
            docs.append(i)
    return docs
