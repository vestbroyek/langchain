from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from dotenv import load_dotenv

# API key
load_dotenv()

if __name__ == "__main__":
    # Grab PDF
    pdf_path = "learning_python.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()

    # Split text into chunks
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )
    docs = text_splitter.split_documents(documents=documents)

    # Generate embeddings
    embeddings = OpenAIEmbeddings()

    # Create in-memory vectorstore
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Optionally, save and load back in
    vectorstore.save_local("faiss_learn_python")
    new_vectorstore = FAISS.load_local("faiss_learn_python", embeddings)

    # Use chain to
    # 1. Embed our query into a vector
    # 2. Send to our vector store and find similar vectors
    # 3. Send our query + context to the LLM
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", retriever=new_vectorstore.as_retriever()
    )
    res = qa.run("Tell me why Python is a good programming language")
    print(res)
