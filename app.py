import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings  
import pickle
import os
from transformers import pipeline 
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

with st.sidebar:
    st.title('Document Chat App')
    st.markdown('''**ABOUT** 
                This project is a chatbot used to discuss anything with your document''')
    
    add_vertical_space(5)
    st.write('Created By © Deepanshu Sharma')

load_dotenv()

def main():
    st.write('Start Your Chat')
    
    pdf = st.file_uploader("Upload your Document (PDF) here", type="pdf")
    
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        
        for page in pdf_reader.pages:
            text += page.extract_text()    
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )    
        chunks = text_splitter.split_text(text=text)
        
        embeddings = HuggingFaceEmbeddings()  # Use HuggingFaceEmbeddings
        store_name = pdf.name[:-4]
        
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                vectorStore = pickle.load(f)
            st.write("Embedding loaded from disk")
        else:        
            vectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(vectorStore, f)
                
            st.write("Embedding completed")
        
        query = st.text_input("Ask question about your Doc")
        
        if query:
            docs = vectorStore.similarity_search(query=query, k=3)
            # st.write(docs)
            
            # llm = OpenAI()
            # chain = load_qa_chain(llm=llm, chain_type="stuff")
            # response = chain.run(input_documents=docs, question=query)
            # st.write(response)
            qa_pipeline = pipeline("question-answering")
            
            # Combine the retrieved docs into a single context
            context = " ".join([doc.page_content for doc in docs])
            
            response = qa_pipeline(question=query, context=context)
            st.write(response['answer'])

if __name__ == '__main__':
    main()
