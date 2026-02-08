import flask_app as app
from langchain_core.messages import HumanMessage, SystemMessage


def retrieve_sources(query, k=20, max_distance=0.9):
    collection = app.chroma_db_client.get_collection(name="pdfs")
    embedded_query = app.embeddings_model.embed_query(query)
    results = collection.query(query_embeddings=[embedded_query], n_results=k)
    filtered_docs = []
    filtered_metas = []

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        if dist < max_distance:
            filtered_docs.append(doc)
            filtered_metas.append(meta)

    results = {
        "documents": filtered_docs,
        "metadatas": filtered_metas,
    }
    print("Retrieved source documents:", results)
    return results


def generate_answer(question):

    sources = retrieve_sources(question)

    context = "\n\n".join(doc for sublist in sources["documents"] for doc in sublist)

    source_files = []

    for d in sources["metadatas"]:
        source_files.append(f"file: {d['source']}, page_num: {d['page']}")

    messages = [
        SystemMessage(
            content="""
                Use semantic similarity to find the answer even if the phrasing differs from the query, 
                If the information is not inferred from the context entirely, respond with "I don't know".
            """
        ),
        HumanMessage(
            content=f"""
                Context:
                {context}

                Question:
                {question}
            """
        ),
    ]

    response = app.openai_client.invoke(messages)

    return response.content, set(source_files)
