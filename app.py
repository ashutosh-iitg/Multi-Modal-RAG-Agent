import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig

import chainlit as cl
from chainlit.types import AskFileResponse

from multiModalRAG import *

welcome_message = """Welcome to the Chainlit Multi-Modal QA demo! To get started:
1. Upload upto 5 files of type PDF, text or image
2. Ask a question about the file
"""
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=int(os.environ.get("CHUNK_SIZE", 1000)),
    chunk_overlap = int(os.environ.get("CHUNK_OVERLAP", 100)),
)

def process_file(file: AskFileResponse):
    Loader = None
    if file.type == "text/plain":
        Loader = TextLoader
    elif file.type == "application/pdf":
        Loader = PyPDFLoader
    else:
        raise ValueError(f"Unsupported file type: {file.type}")
    
    loader = Loader(file.path)
    docs = loader.load()
    texts = [d.page_content for d in docs]
    # texts = text_splitter.split_documents(docs)
    # for i, text in enumerate(texts):
    #     text.metadata["source"] = f"source_{i}"
    return texts


@cl.on_chat_start
async def start():
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            author="assistant",
            content=welcome_message,
            accept=["text/plain", "application/pdf", "image/*"],
            max_files=10,
            max_size_mb=20,
            timeout=300
        ).send()

    if len(files) > 5:
        files = files[:5]

    msg = cl.Message(author="assistant", content=f"Processing {len(files)} files ...")
    await msg.send()

    image_files = []
    texts = []
    tables = []
    for file in files:
        if file.type in ("image/jpeg", "image/png", "image/webp", "image/gif"):
            image_files.append(file.path)
        else:
            texts.extend(process_file(file))

    # Get text and table summaries
    texts = texts[:5] # Limit to 5 pages for demo
    text_summaries, table_summaries = generate_text_summaries(texts, tables, summarize_texts=True)

    # Image summaries
    image_base64_list, image_summaries = generate_image_summaries(image_files)
    
    # The vectorstore to use to index the summaries
    vectorstore = Chroma(
        collection_name="mm_rag",
        embedding_function=VertexAIEmbeddings(model_name="textembedding-gecko@latest"),
    )

    # Create the retriever
    retriever_multi_vector_img = create_multi_vector_retriever(
        vectorstore=vectorstore,
        text_summaries=text_summaries,
        texts=texts,
        table_summaries=table_summaries,
        tables=tables,
        image_summaries=image_summaries,
        images=image_base64_list,
    )

    # Create RAG chain
    chain_multimodal_rag = multi_modal_rag_chain(retriever_multi_vector_img)

    # Let the user know that the system is ready
    msg.content = f"Files processed. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain_multimodal_rag)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  # type: Runnable
    msg = cl.Message(content="", author="assistant")
    
    # chain.astream(message.content, config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]))
    async for token in chain.astream(message.content):
        await msg.stream_token(token)

    await msg.send()


# query = "What are the EV / NTM and NTM rev growth for MongoDB, Cloudflare, and Datadog?"
# docs = retriever_multi_vector_img.get_relevant_documents(query, limit=1)


# TODO: 
# Add a way to get the relevant documents from the retriever
# Add persistence in vectors
# Add a way to detect companies from a query and retrieve vector from the store
# CLiP embeddings