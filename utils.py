import os

import flask_app as app
from pypdf import PdfReader


def load_pdf(file_name):
    documents = []

    if not file_name.lower().endswith(".pdf"):
        return documents

    path = os.path.join(app.UPLOAD_FOLDER, file_name)
    reader = PdfReader(path)

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            documents.append(
                {"text": text, "metadata": {"source": file_name, "page": page_num + 1}}
            )

    return documents
