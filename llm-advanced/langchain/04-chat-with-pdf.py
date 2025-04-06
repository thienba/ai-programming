from langchain_community.document_loaders import PyPDFLoader
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import base64
import gradio as gr
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

def read_pdf_content(pdf_path: str):
    """
    Reads the content of a PDF file and returns a list of documents.
    """
    if pdf_path is None:
        raise ValueError("PDF path is required")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    return docs

def chunk_docs(docs):
    """
    Chunks a list of documents into smaller chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
    chunks = text_splitter.split_documents(docs)
    for chunk in chunks:
        chunk.id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.page_content))
    return chunks

def get_or_create_vector_store(file_path: str):
    """
    Gets or creates a vector store.
    """
    if file_path is None:
        raise ValueError("File path is required")
    file_name = file_path.split("/")[-1]
    vectorstore = Chroma(
        persist_directory="./data",
        collection_name=file_name,
        embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    )
    return vectorstore

def create_pdf_html(file_path: str):
    """Create HTML to preview PDF"""
    if file_path is None:
        return ""

    try:
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')

        pdf_display = f'''
            <div style="width: 100%; height: 800px;">
                <iframe
                    src="data:application/pdf;base64,{base64_pdf}"
                    width="100%"
                    height="100%"
                    style="border: none;">
                </iframe>
            </div>
        '''
        return pdf_display
    except Exception as e:
        return f"Error displaying PDF: {str(e)}"
    
def process_file(file_path: str | None):
    """Process uploaded PDF file and return status with HTML preview"""

    if file_path is None:
        yield "Please upload a PDF file first.", ""
        return  # Add early return to prevent further execution

    try:
        # Create PDF preview
        yield "Reading PDF file...", ""
        pdf_html = create_pdf_html(file_path)
        yield "Processing PDF file...", pdf_html

        # Read PDF content and split into chunks
        documents = read_pdf_content(file_path)
        if not documents:
            yield "Error processing PDF file.", ""
            return

        chunks = chunk_docs(documents)  # Fixed function name from chunk_document to chunk_docs

        # Create vector store and add text chunks
        yield "Creating vector store...", pdf_html
        vectorstore = get_or_create_vector_store(file_path)
        yield "Adding documents to vector store...", pdf_html
        vectorstore.add_documents(chunks)

        yield "PDF file processed successfully.", pdf_html
    except Exception as e:
        yield f"Error processing file: {str(e)}", ""

openai_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.getenv("OPENAI_API_KEY"))

def chat_with_pdf(file_path: str, message: str, history):
    """Process chat interaction"""
    if not message:
        yield []

    try:
        history.append((message, "Processing..."))
        yield "", history

        # Get related data from VectorDB
        vectorstore = get_or_create_vector_store(file_path)
        results = vectorstore.similarity_search(query=message, k=3)

        if not results:
            history.append((message, "Not found data in PDF."))
            yield "", history

        history[-1] = (message, "Found data in VectorDB!")
        yield "", history

        # Bring data into the context of the prompt to answer
        CONTEXT = ""
        for document in results:
            CONTEXT += document.page_content + "\n\n"

        prompt = f"""
        Use the following CONTEXT to answer the QUESTION at the end.
        If you don't know the answer or unsure of the answer, just say that you don't know, don't try to make up an answer.
        Use an unbiased and journalistic tone.

        CONTEXT: {CONTEXT}
        QUESTION: {message}
        """

        print(prompt)

        response = openai_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "user", "content": prompt}
            ],
        )

        # Update the last message pair
        history[-1] = (message, response.choices[0].message.content)
        yield "", history
    except Exception as e:
        print('error', e)
        history.append((message, f"Error: {str(e)}"))
        return "", history

def create_ui():
    """Create Gradio UI"""
    with gr.Blocks() as demo:
        gr.Markdown("# Chat with PDF")

        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="Upload PDF",
                    file_types=[".pdf"],
                )
                process_button = gr.Button("Process PDF")
                status_output = gr.Textbox(label="Status")
                pdf_preview = gr.HTML(label="PDF Preview")

            with gr.Column(scale=1):
                message_box = gr.Textbox(label="Ask a question about your PDF")
                chatbot = gr.Chatbot(height=600)

        # Process events
        process_button.click(
            fn=process_file,
            inputs=[file_input],
            outputs=[status_output, pdf_preview]
        )

        # When the user submits a question, call the chat_with_pdf function
        message_box.submit(
            fn=chat_with_pdf,
            inputs=[file_input, message_box, chatbot],
            outputs=[message_box, chatbot]
        )

    return demo

demo = create_ui()
demo.launch()