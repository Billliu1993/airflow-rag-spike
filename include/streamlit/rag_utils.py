"""
RAG utilities for hybrid search and answer generation.
"""

import os
from typing import List, Dict
import psycopg2
from openai import OpenAI


def generate_embedding(query_text: str, api_key: str = None) -> List[float]:
    """Generate embedding for query using OpenAI."""
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment")
    
    client = OpenAI(api_key=api_key)
    
    # Clean query text
    clean_query = query_text.replace("\n", " ")
    
    # Generate embedding
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=clean_query,
        dimensions=512
    )
    
    return response.data[0].embedding


def vector_search(
    cursor,
    query_embedding: List[float],
    top_k: int = 5
) -> List[Dict]:
    """Perform vector similarity search."""
    embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
    
    query = """
        SELECT 
            id,
            content,
            metadata->>'filename' as filename,
            metadata->>'chunk_index' as chunk_index,
            1 - (embedding <=> %s::vector) as score
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    
    cursor.execute(query, (embedding_str, embedding_str, top_k))
    results = cursor.fetchall()
    
    return [
        {
            "id": r[0],
            "content": r[1],
            "filename": r[2],
            "chunk_index": r[3],
            "score": r[4]
        }
        for r in results
    ]


def bm25_search(
    cursor,
    query_text: str,
    top_k: int = 5
) -> List[Dict]:
    """Perform BM25 keyword search."""
    # Format query for ParadeDB
    terms = query_text.split()
    if len(terms) == 1:
        formatted_query = f"content:{terms[0]}"
    else:
        formatted_query = " OR ".join([f"content:{term}" for term in terms])
    
    query = """
        SELECT 
            id,
            content,
            metadata->>'filename' as filename,
            metadata->>'chunk_index' as chunk_index,
            paradedb.score(id) as score
        FROM documents
        WHERE content @@@ %s
        ORDER BY score DESC
        LIMIT %s
    """
    
    cursor.execute(query, (formatted_query, top_k))
    results = cursor.fetchall()
    
    return [
        {
            "id": r[0],
            "content": r[1],
            "filename": r[2],
            "chunk_index": r[3],
            "score": r[4]
        }
        for r in results
    ]


def reciprocal_rank_fusion(
    vector_results: List[Dict],
    bm25_results: List[Dict],
    k: int = 60,
    vector_weight: float = 0.5,
    bm25_weight: float = 0.5
) -> List[Dict]:
    """Combine results using Reciprocal Rank Fusion."""
    doc_scores: Dict[int, Dict] = {}
    
    # Process vector results
    for rank, result in enumerate(vector_results, 1):
        doc_id = result["id"]
        rrf_score = vector_weight / (k + rank)
        doc_scores[doc_id] = {
            'score': rrf_score,
            'result': result
        }
    
    # Process BM25 results
    for rank, result in enumerate(bm25_results, 1):
        doc_id = result["id"]
        rrf_score = bm25_weight / (k + rank)
        
        if doc_id in doc_scores:
            doc_scores[doc_id]['score'] += rrf_score
        else:
            doc_scores[doc_id] = {
                'score': rrf_score,
                'result': result
            }
    
    # Sort by combined score
    sorted_results = sorted(doc_scores.items(), key=lambda x: x[1]['score'], reverse=True)
    
    # Return results with combined score
    final_results = []
    for doc_id, data in sorted_results:
        result = data['result'].copy()
        result['score'] = data['score']
        final_results.append(result)
    
    return final_results


def hybrid_search(
    query: str,
    mode: str = "hybrid",
    top_k: int = 5,
    db_host: str = "localhost",
    db_port: int = 5432,
    db_name: str = "vector_db",
    db_user: str = "postgres",
    db_password: str = "postgres"
) -> List[Dict]:
    """
    Perform hybrid search combining vector and BM25.
    
    Args:
        query: Search query
        mode: "vector", "bm25", or "hybrid"
        top_k: Number of results to return
        db_host: Database host
        db_port: Database port
        db_name: Database name
        db_user: Database user
        db_password: Database password
    
    Returns:
        List of search results with content, filename, chunk_index, and score
    """
    # Connect to database
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password
    )
    cursor = conn.cursor()
    
    try:
        if mode == "vector":
            query_embedding = generate_embedding(query)
            results = vector_search(cursor, query_embedding, top_k)
        
        elif mode == "bm25":
            results = bm25_search(cursor, query, top_k)
        
        elif mode == "hybrid":
            query_embedding = generate_embedding(query)
            
            # Fetch more results for better fusion
            fetch_k = top_k * 2
            vector_results = vector_search(cursor, query_embedding, fetch_k)
            bm25_results = bm25_search(cursor, query, fetch_k)
            
            # Combine using RRF
            results = reciprocal_rank_fusion(vector_results, bm25_results)[:top_k]
        
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        return results
    
    finally:
        cursor.close()
        conn.close()


def generate_answer(
    query: str,
    search_results: List[Dict],
    model: str = "gpt-4o-mini",
    api_key: str = None
) -> str:
    """
    Generate an answer using LLM with retrieved context.
    
    Args:
        query: User's question
        search_results: Results from hybrid search
        model: OpenAI model to use
        api_key: OpenAI API key
    
    Returns:
        Generated answer
    """
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment")
    
    client = OpenAI(api_key=api_key)
    
    # Build context from search results
    context_parts = []
    for i, result in enumerate(search_results, 1):
        context_parts.append(
            f"[Source {i} - {result['filename']}, Chunk {result['chunk_index']}]\n"
            f"{result['content']}\n"
        )
    
    context = "\n\n".join(context_parts)
    
    # Create prompt
    system_prompt = """You are a helpful financial analyst assistant. Answer questions based on the provided context from financial documents.

Guidelines:
- Use information from the provided context to answer the question
- If the context doesn't contain enough information, say so clearly
- Cite specific sources when possible (e.g., "According to Source 1...")
- Be precise with numbers, dates, and financial figures
- If multiple sources provide conflicting information, acknowledge it
- Keep answers concise but comprehensive"""
    
    user_prompt = f"""Context from documents:

{context}

---

Question: {query}

Please provide a clear, accurate answer based on the context above."""
    
    # Generate response
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
        max_tokens=1000
    )
    
    return response.choices[0].message.content
