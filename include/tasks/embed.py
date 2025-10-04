"""
Tasks for generating embeddings from document chunks using OpenAI.
"""

import logging
import os
from typing import List, Dict

from openai import OpenAI

logger = logging.getLogger(__name__)


def generate_embeddings(
    chunks: List[Dict[str, any]],
    model_name: str = "text-embedding-3-small",
    embedding_dimensions: int = 512,
    api_key: str = None,
) -> List[Dict[str, any]]:
    """
    Generate embeddings for document chunks using OpenAI API.
    
    Args:
        chunks: List of document chunks from the chunking task
        model_name: OpenAI model name (default: text-embedding-3-small)
        embedding_dimensions: Dimension of embeddings (default: 512)
        api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        
    Returns:
        List of chunks with embeddings added
    """
    if not chunks:
        logger.info("No chunks to embed. Skipping.")
        return []
    
    logger.info(f"Starting embedding generation for {len(chunks)} chunks...")
    logger.info(f"Using OpenAI model: {model_name} with {embedding_dimensions} dimensions")
    
    # Initialize OpenAI client
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
    
    client = OpenAI(api_key=api_key)
    
    embedded_chunks = []
    
    # Process chunks individually for better error handling and progress tracking
    for i, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {i + 1}/{len(chunks)}")
        
        try:
            # Clean text: replace newlines with spaces for better embedding quality
            text = chunk["content"].replace("\n", " ")
            
            # Generate embedding using OpenAI API
            response = client.embeddings.create(
                model=model_name,
                input=text,
                dimensions=embedding_dimensions
            )
            
            # Extract embedding
            embedding = response.data[0].embedding
            
            # Add embedding to chunk
            embedded_chunk = chunk.copy()
            embedded_chunk["embedding"] = embedding
            embedded_chunks.append(embedded_chunk)
            logger.info(f"Successfully embedded chunk {i + 1}/{len(chunks)}")
                
        except Exception as e:
            logger.error(f"Error embedding chunk {i + 1}: {str(e)}")
            # Add chunk without embedding to preserve data
            embedded_chunk = chunk.copy()
            embedded_chunk["embedding"] = None
            embedded_chunks.append(embedded_chunk)
            continue
    
    successful_embeddings = sum(1 for c in embedded_chunks if c.get("embedding") is not None)
    logger.info(f"Successfully generated {successful_embeddings} embeddings out of {len(chunks)} chunks")
    
    # Log embedding dimensions
    if embedded_chunks and embedded_chunks[0].get("embedding"):
        embedding_dim = len(embedded_chunks[0]["embedding"])
        logger.info(f"Embedding dimension: {embedding_dim}")
    
    return embedded_chunks

