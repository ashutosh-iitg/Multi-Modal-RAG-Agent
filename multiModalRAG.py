import os
import io
import re
import uuid
import base64
from PIL import Image

import vertexai

from langchain_community.document_loaders import PyPDFLoader
from langchain_google_vertexai import VertexAI, ChatVertexAI, VertexAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_core.documents import Document

from langchain_core.runnables import RunnableLambda, RunnablePassthrough

import google.auth

import logging

logging.basicConfig(level=logging.INFO)

credentials, _ = google.auth.default()

PROJECT_ID = os.environ.get('PROJECT_ID')
REGION = os.environ.get('REGION')
VERTEX_API_KEY = os.environ.get('VERTEX_API_KEY')

vertexai.init(project=PROJECT_ID, location=REGION, credentials=credentials)

def generate_text_summaries(texts, tables, summarize_texts=False):
    """
    Generates text and table summaries for retrieval.
    Args:
        texts (list): List of text elements to be summarized.
        tables (list): List of table elements to be summarized.
        summarize_texts (bool, optional): Flag indicating whether to summarize texts. Defaults to False.
    Returns:
        tuple: A tuple containing the text summaries and table summaries.
    """

    
    # Prompt
    prompt_text = """You are an assistant tasked with summarizing tables and texts for retrieval. \
        These summaries will be embedded and used to retrieve the raw text or table elements. \
        Give a concise summary of the table or text that is well optimized for retrieval. Table or text: {element} """
    prompt = PromptTemplate.from_template(prompt_text)
    empty_response = RunnableLambda(
        lambda x: AIMessage(content="Error processing Document.")
    )

    # Text summary chain
    model = VertexAI(
        temperature=0,
        model_name="gemini-pro",
        max_output_tokens = 1024
    ).with_fallbacks([empty_response])

    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    # Initialize empty summaries
    text_summaries = []
    table_summaries = []

    # Apply to text if texts are provided and summarization is requested
    if texts and summarize_texts:
        text_summaries = summarize_chain.batch(texts, {"max_concurrency": 1})
    elif texts:
        text_summaries = texts

    # Apply to tables if tables are provided
    if tables:
        table_summaries = summarize_chain.batch(tables, {"max_concurrency": 1})

    return text_summaries, table_summaries


def encode_image(image_path):
    """
    Encodes the image located at the given image_path into a base64 encoded string.
    Parameters:
    image_path (str): The path to the image file.
    Returns:
    str: The base64 encoded string representation of the image.
    """
    
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string


def image_summarize(image_base64, prompt):
    """
    Summarizes an image using a pre-trained Gemini Pro Vision model.
    Args:
        image_base64 (str): The base64 encoded image.
        prompt (str): The prompt for generating the image summary.
    Returns:
        str: The generated image summary.
    """

    model = ChatVertexAI(model_name="gemini-pro-vision", max_output_tokens=1024)

    msg = model(
        [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                    },
                ]
            )
        ]
    )

    return msg.content


def generate_image_summaries(image_files:list[str]):
    """
    Generates image summaries for retrieval.
    Args:
        image_files (list): List of image files to be summarized.
    Returns:
        tuple: A tuple containing two lists. The first list contains the base64 encoded images, 
        and the second list contains the generated image summaries.
    """

    # Store base64 encoded images
    image_base64_list = []

    # Store image summaries
    image_summaries = []

    # Prompt
    prompt = """You are an assistant tasked with summarizing images for retrieval. \
        These summaries will be embedded and used to retrieve the raw image. \
        Give a concise summary of the image that is well optimized for retrieval."""
    
    # Apply to images
    for image_path in sorted(image_files):
        image_base64 = encode_image(image_path)
        image_base64_list.append(image_base64)
        image_summaries.append(image_summarize(image_base64, prompt))
        
    return image_base64_list, image_summaries


def create_multi_vector_retriever(
        vectorstore, text_summaries, texts, table_summaries, tables, image_summaries, images
):
    """
    Create a multi-vector retriever.
    Args:
        vectorstore: The vectorstore used for vector representation.
        text_summaries: List of text summaries.
        texts: List of text contents.
        table_summaries: List of table summaries.
        tables: List of table contents.
        image_summaries: List of image summaries.
        images: List of image contents.
    Returns:
        retriever: The created multi-vector retriever.
    """
    
    # Initialize the storage layer
    store = InMemoryStore()
    id_key = "doc_id"

    # Create the multi-vector retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    # Helper function to add documents to the vectorstore and docstore
    def add_documents(retriever, doc_summaries, doc_contents):
        """
        Adds documents to the retriever's vectorstore and docstore.
        Args:
            retriever (Retriever): The retriever object.
            doc_summaries (List[str]): List of document summaries.
            doc_contents (List[str]): List of document contents.
        Returns:
            None
        """
        doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]}) for i, s in enumerate(doc_summaries)
        ]
        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, doc_contents)))

    # Add texts, tables and images
    # Check if text summaries are available
    if text_summaries:
        add_documents(retriever, text_summaries, texts)

    # Check if table summaries are available
    if table_summaries:
        add_documents(retriever, table_summaries, tables)

    # Check if image summaries are available
    if image_summaries:
        add_documents(retriever, image_summaries, images)

    return retriever


def is_base64(sb):
    """
    Check if the given string is a base64 encoded string.
    Args:
        sb (str): The string to check.
    Returns:
        bool: True if the string is base64 encoded, False otherwise.
    """
    
    # Check if the string looks like base64
    return re.match(r'^[A-Za-z0-9+/]+[=]{0,2}$', sb) is not None


def is_image_data(b64data):
    """
    Check if the given base64 encoded string represents an image.
    Args:
        b64data (str): The base64 encoded string.
    Returns:
        bool: True if the string represents an image, False otherwise.
    """
    
    # Check if the string starts with the image data prefix
    image_signatures = {
        b"\xFF\xD8\xFF": "jpg",
        b"\xFF\xD8\xFF\xE0": "jpeg",
        b"\x89\x50\x4E\x47\x0D\x0A\x1A\x0A": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    
    try:
        header = base64.b64decode(b64data)[:8]     # Decode and Get the first 8 bytes
        for sig, format in image_signatures.items():
            if header.startswith(sig):
                return True
            return False
    except Exception:
        return False
    

def resize_base64_image(base64_str, size=(224, 224)):
    """
    Resize the image represented by the base64 encoded string to the given size.
    Args:
        base64_str (str): The base64 encoded string representing the image.
        size (tuple, optional): The size to resize the image to. Defaults to (400, 400).
    Returns:
        str: The base64 encoded string representing the resized image.
    """
    
    # Decode the base64 string into a PIL image
    image = Image.open(io.BytesIO(base64.b64decode(base64_str)))
    format = image.format
    
    # Resize the image
    image = image.resize(size, Image.LANCZOS)
    
    # Encode the resized image back to base64
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def split_image_text_types(docs):
    """
    Split the given documents into image and text types.
    Args:
        docs (list): The list of documents to split.
    Returns:
        dict: A dictionary containing the image and text type documents.
    """
    
    # Initialize the image and text type documents
    image_docs = []
    text_docs = []
    
    # Check if the document is an image or text type
    for doc in docs:
        # Check if the document is a Document object
        if isinstance(doc, Document):
            doc = doc.page_content

        # Check if the document is base64 encoded and represents an image
        if is_base64(doc) and is_image_data(doc):
            doc = resize_base64_image(doc, size=(1300, 600))
            image_docs.append(doc)
        else:
            text_docs.append(doc)

    if len(image_docs) > 0:
        return {"images": image_docs[:1], "texts": []} # Only return the first image if there are multiple images
    
    return {"images": image_docs, "texts": text_docs}


def image_prompt_func(data_dict):
    """
    Generates a list of messages for a financial analyst to provide investment advice based on given data.
    Parameters:
    - data_dict (dict): A dictionary containing the following keys:
        - "context" (dict): A dictionary containing the following keys:
            - "texts" (list): A list of strings representing the context texts.
            - "images" (list): A list of image data in base64 format.
        - "question" (str): The user's question.
    Returns:
    - list: A list of messages to be sent as a response, including text and image messages.
    """

    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []

    # Adding the text for analysis
    text_message = {
        "type": "text",
        "text": (
            "You are a financial analyst tasked with providing investment advice. \n"
            "You will be given a mixed of text, tables, and image(s) usually of charts or graphs. \n"
            "Use this information to provide investment advice related to the user question. \n"
            f"User-provided question: {data_dict['question']}\n\n"
            "Text and / or tables:\n"
            f"{formatted_texts}"
        ),
    }

    messages.append(text_message)

    # Adding image(s) to the messages if present
    if data_dict["context"]["images"]:
        for image in data_dict["context"]["images"]:
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
            messages.append(image_message)
    return [HumanMessage(content=messages)]


def multi_modal_rag_chain(retriever):
    """
    Constructs a RAG (Retrieval-Augmented Generation) pipeline for multi-modal inputs.
    Args:
        retriever: The retriever component used to retrieve relevant information.
    Returns:
        The RAG pipeline for multi-modal inputs.
    """

    # Multi-modal LLM
    model = ChatVertexAI(
        temperature=0,
        model_name="gemini-pro-vision",
        max_output_tokens=1024,
    )

    # RAG Pipeline
    chain = (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(image_prompt_func)
        | model
        | StrOutputParser()
    )

    return chain