from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
import streamlit as st
from io import BytesIO
import fitz
import os
from langchain.schema import Document
import qdrant_client 
load_dotenv()

# Define constants for Qdrant connection
if 'QDRANT_URL' in st.secrets:
    # Use secrets from Streamlit Cloud
    QDRANT_URL = st.secrets["QDRANT_URL"]
    QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
else:
    # Use environment variables from .env file for local development
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

qdrant_url =QDRANT_URL
qdrant_key= QDRANT_API_KEY
QDRANT_COLLECTION_NAME="PDF_Rag_Agent"

@st.cache_resource  
def indexing(file):
    

    #To delete the old collection from vector DB
    try:
        # 1. To create the qrdant client to interact with the DB
        qdrant_client_instance=qdrant_client.QdrantClient(url=qdrant_url,api_key=qdrant_key)

        # 2. To deleting the old collection
        # st.write(f"Deleting the old collection:" '{QDRANT_COLLECTION_NAME}')
        qdrant_client_instance.delete_collection(collection_name=QDRANT_COLLECTION_NAME)
        # st.write("Collection deleted successfully.")
    except Exception as e:
        # This will likely happen on the very first run when the collection doesn't exist yet.
        # It's safe to ignore this specific error.
        st.write(f"Collection did not exist or another error occurred: {e}")







    #Step 1 Loading of the file.
    pdf_bytes=file.read()

    pdf_stream=BytesIO(pdf_bytes)

    docs=fitz.open(stream=pdf_stream)
    pages=[]
    for i,page in enumerate(docs):
        text=page.get_text() 
        pages.append(Document(page_content=text,metadata={"page":i+1}))
    #Why I did this because in langcahi for splitting purpose we need lanchain documents and we cannot pass string. thats why  "Document" helps to do this
        

    #step 2 Chunking of the file
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000, #Dividing the whole data into small small sets (in this case each set will have 1000 caracters)
        chunk_overlap=400 # (chunk_overlap means to get some context of the previous set as well)
    )
    split_docs=text_splitter.split_documents(documents=pages)

    # st.write(split_docs)

    #step 3 vector embeddings of the chunks
    
    embedding_model=OpenAIEmbeddings(
        model="text-embedding-3-large"
    )
    #Using embedding_model create embeddings of split_docs and store in the vector_db
    QdrantVectorStore.from_documents(
        documents=split_docs,
        url=qdrant_url,
        api_key=qdrant_key,
        collection_name="PDF_Rag_Agent",
        embedding=embedding_model
    )

    


   