import os
import flask_app as app
from pypdf import PdfReader


def load_pdfs_from_dir(pdf_dir):
    documents = []

    for filename in os.listdir(pdf_dir):
        if not filename.lower().endswith(".pdf"):
            continue

        path = os.path.join(pdf_dir, filename)
        reader = PdfReader(path)

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                documents.append(
                    {"text": text, "metadata": {"source": filename, "page": page_num + 1}}
                )

    return documents


def chunk_documents(docs, chunk_size=600, overlap=100):
    chunks = []

    for doc in docs:
        text = doc["text"]
        meta = doc["metadata"]

        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            chunks.append({"text": chunk_text, "metadata": meta})

            start = end - overlap

    return chunks


def embedding_chunks(chunks):
    texts = [c["text"] for c in chunks]
    embeddings = app.embeddings_model.embed_documents(texts)
    return embeddings


def initialize_chromadb():

    pdf_documents = load_pdfs_from_dir("data/")
    collection = app.chroma_db_client.get_or_create_collection(name="pdfs")
    chunks = chunk_documents(pdf_documents)
    embeddings = embedding_chunks(chunks)

    ids = [
        f'{c["metadata"]["source"]}_page_{c["metadata"]["page"]}_chunk_{i}'
        for i, c in enumerate(chunks)
    ]

    collection.add(
        documents=[c["text"] for c in chunks],
        embeddings=embeddings,
        metadatas=[c["metadata"] for c in chunks],
        ids=ids,
    )

    return
