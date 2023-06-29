from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os

# API key
load_dotenv()

# Pinecone init
pinecone.init(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environment=os.environ.get("PINECONE_ENVIRONMENT_REGION"),
)


def ingest_docs() -> None:
    loader = ReadTheDocsLoader(
        path="rtdocs/api.python.langchain.com/en/latest",
        encoding="ISO-8859-1",
        features="lxml",
    )
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(documents=raw_documents)

    # clean up metadata
    for doc in documents:
        old_path = doc.metadata["source"]
        new_url = old_path.replace("rtdocs/", "https://")
        doc.metadata.update({"source": new_url})

    print(f"Going to insert {len(documents)} into Pinecone")

    # generate embeddings
    embeddings = OpenAIEmbeddings()
    Pinecone.from_documents(documents, embeddings, index_name="langchain-doc-index")


if __name__ == "__main__":
    ingest_docs()
