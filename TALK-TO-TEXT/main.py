from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# ✅ Paths
DATA_PATH = os.path.expanduser("~/Desktop/data for rag")  # Your folder with TXT files
DB_FAISS_PATH = "vectorstore/db_faiss"

# 📂 Load .txt files
def load_txt_files(data):
    loader = DirectoryLoader(data, glob="*.txt", loader_cls=TextLoader)
    return loader.load()

# ✂️ Split into better chunks
def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # larger context
        chunk_overlap=200 # keeps flow between chunks
    )
    return splitter.split_documents(documents)

# 🧠 Use better embeddings for semantic search
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# 💾 Save to FAISS
def save_to_faiss(chunks, model):
    db = FAISS.from_documents(chunks, model)
    db.save_local(DB_FAISS_PATH)
    print(f"✅ FAISS DB saved at: {DB_FAISS_PATH}")

# ▶️ Run pipeline
if __name__ == "__main__":
    os.makedirs("vectorstore", exist_ok=True)
    docs = load_txt_files(DATA_PATH)
    print(f"📄 Loaded {len(docs)} documents")
    chunks = create_chunks(docs)
    print(f"✂️ Created {len(chunks)} text chunks")
    model = get_embedding_model()
    save_to_faiss(chunks, model)
