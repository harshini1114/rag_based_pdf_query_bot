import os
from chromadb.config import Settings
from flask import Flask, jsonify, render_template, request
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from werkzeug.utils import secure_filename

import chromadb

import intialize_chromadb
import my_agent
import utils


app = Flask(__name__)
app.debug = True
app.secret_key = "supersecretkey"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

openai_client = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=OPENAI_API_KEY)

chroma_db_client = chromadb.PersistentClient(
    path=".chroma_db/", settings=Settings(allow_reset=True)
)  # ignore

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}

app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB limit
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure the upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def hello_page():
    return render_template("start_page.html")


@app.route("/chat", methods=["POST"])
def chat_page():
    data = request.get_json()
    question = data.get("question")

    if not question:
        return jsonify({"error": "Question is required"}), 400

    answer, source_files = my_agent.generate_answer(question)

    print("Sources used:", source_files)

    if answer == "I don't know.":
        source_files = []

    return jsonify({"question": question, "answer": answer, "sources": list(source_files)})


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"message": "No file part"})

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"message": "No selected file"})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(UPLOAD_FOLDER, filename))
        documents = utils.load_pdf(filename)
        collection = chroma_db_client.get_or_create_collection(name="pdfs")
        chunks = intialize_chromadb.chunk_documents(documents)
        embeddings = intialize_chromadb.embedding_chunks(chunks)
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
        return jsonify({"message": f"File {filename} uploaded and processed successfully!"})

    else:
        return jsonify({"message": "Invalid file type"})


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File is too large. Max limit is 5MB"}), 413


if __name__ == "__main__":
    chroma_db_client.reset()  # Clear existing database for fresh start
    intialize_chromadb.initialize_chromadb()

    app.run()
