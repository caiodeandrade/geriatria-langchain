import os
import glob
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Inicializa o aplicativo Flask, responsável por receber requisições HTTP
app = Flask(__name__)

# Define o diretório dos PDFs de origem e diretório do índice FAISS
pdfs_dir = "assets"
faiss_path = "pdf_faiss_index"


# Função para carregar um índice vetorial já existente ou criar um novo a partir dos PDFs
def load_or_create_faiss_index():
    # Se já existe um índice FAISS salvo, carrega do disco
    if os.path.exists(faiss_path):
        print("Usou atual")
        return FAISS.load_local(
            faiss_path,
            OpenAIEmbeddings(openai_api_key=openai_api_key),
            allow_dangerous_deserialization=True,
        )
    else:
        print("Criou novo")
        # Caso não exista, processa todos os PDFs da pasta 'assets'
        pdf_files = glob.glob(os.path.join(pdfs_dir, "*.pdf"))
        if not pdf_files:
            raise FileNotFoundError("Nenhum arquivo PDF encontrado na pasta 'assets'.")

        # 1. Carrega todos os PDFs como lista de documentos
        docs = []
        for pdf_file in pdf_files:
            loader = PyPDFLoader(pdf_file)
            docs.extend(loader.load())

        # 2. Divide o texto em blocos menores (chunking)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False
        )
        split_docs = text_splitter.split_documents(docs)

        # 3. Vetoriza os blocos usando embeddings do OpenAI
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

        # 4. Cria e salva o banco vetorial FAISS localmente
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        vectorstore.save_local(faiss_path)
        return vectorstore


# Carrega ou cria o banco vetorial FAISS ao iniciar o servidor
vectorstore = load_or_create_faiss_index()

# Cria a cadeia de perguntas e respostas (RAG) conectando o modelo LLM ao retriever
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(openai_api_key=openai_api_key),
    retriever=vectorstore.as_retriever()
)

# Define o endpoint da API para receber perguntas (POST /query)
@app.route("/query", methods=["POST"])
def query():
    data = request.json
    question = data.get("question")  # Extrai a pergunta do corpo da requisição
    if not question:
        return jsonify({"error": "Missing 'question' in request body"}), 400
    answer = qa_chain.run(question)
    return jsonify({"answer": answer})


# Inicia o servidor Flask se o script for executado diretamente
if __name__ == "__main__":
    app.run(debug=True)