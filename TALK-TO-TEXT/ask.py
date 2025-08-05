import os
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# 🔐 Make sure your token is set in env
HF_TOKEN = os.environ.get("HF_Thf_PVEhduTiQpyWXSseWmOtgFMKmnZbSULrCkOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don’t know the answer, say "I don't know". Don’t make anything up.

Context: {context}
Question: {question}

Answer:
"""

def load_llm():
    return HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        temperature=0.5,
        model_kwargs={
            "token": HF_TOKEN,
            "max_length": "512"
        }
    )

def set_custom_prompt():
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

def main():
    print("🔗 Loading vector DB...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("vectorstore/db_faiss", embedding_model, allow_dangerous_deserialization=True)

    print("💬 Initializing QA chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=load_llm(),
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": set_custom_prompt()}
    )

    user_query = input("🧠 Ask your question: ")
    response = qa_chain.invoke({"query": user_query})

    print("\n📌 RESULT:\n", response["result"])
    print("\n📚 SOURCE DOCUMENTS:\n", response["source_documents"])

if __name__ == "__main__":
    main()


