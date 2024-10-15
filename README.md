# Document Chat App

This project is a **Document Chat Application** that allows users to upload a PDF document and ask questions about its content. The app uses **Google Generative AI** and **LangChain** to extract and provide relevant answers from the uploaded document.

## Features

- Upload a PDF file.
- Extracts text from the uploaded document.
- Splits the extracted text into smaller chunks for processing.
- Embeds the text using **HuggingFaceEmbeddings**.
- Stores and retrieves embeddings using **FAISS** vector store.
- Uses **Google Generative AI** to answer questions based on the document context.

## Tech Stack

- **Streamlit**: Frontend for file upload and chat functionality.
- **LangChain**: Manages the text processing, embeddings, and interaction with AI models.
- **Google Generative AI**: Provides advanced language model capabilities to answer user queries.
- **FAISS**: For similarity search and efficient vector storage.
- **HuggingFaceEmbeddings**: Converts text into embeddings for further analysis.

## Installation

To get started with the project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/deepanshu452/PDF-Q-A.git
