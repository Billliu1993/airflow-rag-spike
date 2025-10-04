"""
Tasks for chunking documents into smaller pieces for embedding.
"""

import logging
from typing import List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def chunk_documents(
    extracted_docs: List[Dict[str, str]],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Dict[str, any]]:
    """
    Chunk documents into smaller pieces using recursive character text splitter.
    
    Args:
        extracted_docs: List of extracted documents from the extract task
        chunk_size: Size of each chunk in characters (default: 500)
        chunk_overlap: Overlap between chunks in characters (default: 50)
        
    Returns:
        List of document chunks with metadata
    """
    if not extracted_docs:
        logger.info("No documents to chunk. Skipping.")
        return []
    
    logger.info(f"Starting chunking for {len(extracted_docs)} documents...")
    logger.info(f"Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
    
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    
    all_chunks = []
    
    for doc in extracted_docs:
        filepath = doc.get("filepath", "unknown")
        filename = doc.get("filename", "unknown")
        content = doc.get("content", "")
        
        if not content:
            logger.warning(f"Empty content for {filename}, skipping...")
            continue
        
        logger.info(f"Chunking {filename}...")
        
        # Split the document content into chunks
        chunks = text_splitter.split_text(content)
        
        logger.info(f"Created {len(chunks)} chunks from {filename}")
        
        # Create chunk objects with metadata
        for idx, chunk_text in enumerate(chunks):
            chunk = {
                "content": chunk_text,
                "metadata": {
                    "filepath": filepath,
                    "filename": filename,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "file_size": doc.get("file_size"),
                    "file_extension": doc.get("file_extension"),
                },
            }
            all_chunks.append(chunk)
    
    logger.info(f"Successfully created {len(all_chunks)} total chunks from {len(extracted_docs)} documents")
    
    return all_chunks

