import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader  # Alterado de DoclingLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# 1. Carregar variáveis de ambiente do arquivo .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# 2. Função para processar o PDF e gerar índice FAISS
def process_pdf_to_faiss(pdf_path):
    # Carregar o PDF usando PyPDFLoader
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Fazer chunking do texto
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False
    )
    split_docs = text_splitter.split_documents(docs)

    # Criar embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Criar e salvar no banco de vetores FAISS
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local("pdf_faiss_index")

    return vectorstore

# 3. Exemplo de uso
pdf_path = "assets/Fundamentos_de_geriatria_19_115.pdf"
faiss_store = process_pdf_to_faiss(pdf_path)

# 4. Buscar similaridade
query = "Qual é o tema principal do documento?"
results = faiss_store.similarity_search(query, k=3)

for result in results:
    print("retorno da busca\n\n")
    print(result.page_content)

