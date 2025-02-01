import streamlit as st
import logging as log
from langchain_community.document_loaders import PyPDFLoader, UnstructuredCSVLoader, UnstructuredExcelLoader, TextLoader, UnstructuredWordDocumentLoader, UnstructuredHTMLLoader, UnstructuredPowerPointLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain, RetrievalQAWithSourcesChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os, tempfile, nltk
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

load_dotenv("config.env")

log.basicConfig(level=log.INFO)

# huggingface_api_key = os.getenv("HUGGINGFACEHUB_API_KEY")

# llm = HuggingFaceHub(
#         repo_id= os.getenv("HF_MODEL"),
#         model_kwargs={"temperature": 0.1, "max_length": 512},
#         huggingfacehub_api_token=huggingface_api_key
#     )

google_api_key = st.secrets["API_KEYS"]["GOOGLE_API_KEY"]
model_name = st.secrets["MODELS"]["GOOGLE_MODEL"]
embedding_model_name = st.secrets["MODELS"]["HF_EMBEDDING"]
faiss_index_name = st.secrets["FAISS_INDEX"]["NAME"]

llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.1,
            max_tokens=512,
            timeout=10,
            max_retries=3,
            google_api_key=google_api_key
        )

embeddings = HuggingFaceEmbeddings(
                model_name = embedding_model_name,
                model_kwargs = {'device': 'cpu'},
                encode_kwargs = {'normalize_embeddings': False}
            )

def loader(files):
    loaders_map = {
        "pdf": (".pdf", PyPDFLoader),
        "xlsx": (".xlsx", UnstructuredExcelLoader),
        "csv": (".csv", UnstructuredCSVLoader),
        "docx": (".docx", UnstructuredWordDocumentLoader),
        "doc": (".docx", UnstructuredWordDocumentLoader),
        "txt": (".txt", TextLoader),
        "pptx": (".pptx", UnstructuredPowerPointLoader),
        "html": (".html", UnstructuredHTMLLoader),
    }
    
    data = []
    
    for file in files:
        file_type = file.name.split(".")[-1]
        
        if file_type not in loaders_map:
            print(f"Unsupported file type : {file_type}, \nPlease upload file type with .pdf, .xlsx, .csv, .txt, .docx, .pptx, .html")
            continue

        suffix, loader_class = loaders_map[file_type]
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name
            
            loader = loader_class(temp_file_path)
            data.extend(loader.load())
        
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    return data

def embed_files(files):
    with st.spinner("Embedding the uploaded files"):
        data = loader(files)
        getVectorStore(data)

def getVectorStore(data):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    documents = splitter.split_documents(data)

    db = FAISS.from_documents(documents, embeddings)
    db.save_local(faiss_index_name)

def getResponse(query, chat_history):

    vector_store = FAISS.load_local(faiss_index_name, embeddings, allow_dangerous_deserialization=True)
    
    with st.spinner("Getting response from the model"):
        condense_question_system_template = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        condense_question_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", condense_question_system_template),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            llm, vector_store.as_retriever(), condense_question_prompt
        )

        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context and given chat history "
            "to answer the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise. "
            "\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
            ]
        )
        qa_chain = create_stuff_documents_chain(llm, qa_prompt)

        convo_qa_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

        result =  convo_qa_chain.invoke(
                    {
                        "input": query,
                        "chat_history": chat_history,
                    }
                )
        
        print("\nModels output - ",result)
        
        # To print the sources
        docs = result["context"]
        for doc in docs:
            log.info(doc.metadata["source"])
        
        return result['answer']
