"""
Test similarity search in pgvector database with hybrid retrieval.

Supports three search modes:
- vector: Pure vector similarity search
- bm25: Pure BM25 keyword search
- hybrid: Combined vector + BM25 using Reciprocal Rank Fusion (RRF)

Usage:
    python test_similarity_search.py "your query here" [mode] [top_k]
    
Examples:
    python test_similarity_search.py "financial risk analysis"
    python test_similarity_search.py "financial risk analysis" hybrid
    python test_similarity_search.py "financial risk analysis" bm25 10
"""

import sys
import os
from openai import OpenAI
import psycopg2
from typing import List, Tuple, Dict


def generate_embedding(query_text: str) -> List[float]:
    """Generate embedding for query using OpenAI."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    print("⚡ Generating query embedding...")
    client = OpenAI(api_key=api_key)
    
    # Clean query text
    clean_query = query_text.replace("\n", " ")
    
    # Generate embedding
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=clean_query,
        dimensions=512
    )
    embedding = response.data[0].embedding
    print(f"✅ Generated {len(embedding)}-dimensional embedding\n")
    return embedding


def vector_search(cursor, query_embedding: List[float], top_k: int = 5) -> List[Tuple]:
    """Perform pure vector similarity search."""
    print("🔎 Performing vector similarity search...\n")
    
    # Convert embedding to string format for pgvector
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
    return cursor.fetchall()


def bm25_search(cursor, query_text: str, top_k: int = 5) -> List[Tuple]:
    """Perform pure BM25 keyword search using ParadeDB."""
    print("📝 Performing BM25 keyword search...\n")
    
    # Format query for ParadeDB - use column:term pairs
    # Split query into terms and prefix each with "content:"
    terms = query_text.split()
    if len(terms) == 1:
        # Single term - simple search
        formatted_query = f"content:{terms[0]}"
    else:
        # Multiple terms - use OR to match any term
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
    return cursor.fetchall()


def reciprocal_rank_fusion(
    vector_results: List[Tuple],
    bm25_results: List[Tuple],
    k: int = 60,
    vector_weight: float = 0.5,
    bm25_weight: float = 0.5
) -> List[Tuple]:
    """
    Combine results using Reciprocal Rank Fusion (RRF).
    
    Args:
        vector_results: Results from vector search
        bm25_results: Results from BM25 search
        k: RRF constant (default 60)
        vector_weight: Weight for vector search (default 0.5)
        bm25_weight: Weight for BM25 search (default 0.5)
    
    Returns:
        Combined and re-ranked results
    """
    print("🔀 Combining results using Reciprocal Rank Fusion...\n")
    
    # Create a mapping of doc_id to (rank, result)
    doc_scores: Dict[int, Dict] = {}
    
    # Process vector results
    for rank, result in enumerate(vector_results, 1):
        doc_id = result[0]
        rrf_score = vector_weight / (k + rank)
        doc_scores[doc_id] = {
            'score': rrf_score,
            'result': result,
            'vector_rank': rank,
            'bm25_rank': None
        }
    
    # Process BM25 results
    for rank, result in enumerate(bm25_results, 1):
        doc_id = result[0]
        rrf_score = bm25_weight / (k + rank)
        
        if doc_id in doc_scores:
            # Document found in both searches - add scores
            doc_scores[doc_id]['score'] += rrf_score
            doc_scores[doc_id]['bm25_rank'] = rank
        else:
            # Document only in BM25 search
            doc_scores[doc_id] = {
                'score': rrf_score,
                'result': result,
                'vector_rank': None,
                'bm25_rank': rank
            }
    
    # Sort by combined score
    sorted_results = sorted(doc_scores.items(), key=lambda x: x[1]['score'], reverse=True)
    
    # Return results with combined score
    final_results = []
    for doc_id, data in sorted_results:
        result = data['result']
        combined_score = data['score']
        # Replace original score with combined score
        final_result = (result[0], result[1], result[2], result[3], combined_score)
        final_results.append(final_result)
    
    return final_results


def similarity_search(
    query_text: str, 
    mode: str = "hybrid", 
    top_k: int = 5,
    vector_weight: float = 0.5,
    bm25_weight: float = 0.5
):
    """
    Perform similarity search in pgvector database.
    
    Args:
        query_text: The search query
        mode: Search mode - "vector", "bm25", or "hybrid" (default: "hybrid")
        top_k: Number of results to return (default: 5)
        vector_weight: Weight for vector search in hybrid mode (default: 0.5)
        bm25_weight: Weight for BM25 search in hybrid mode (default: 0.5)
    """
    mode = mode.lower()
    if mode not in ["vector", "bm25", "hybrid"]:
        print(f"❌ Invalid mode: {mode}. Must be 'vector', 'bm25', or 'hybrid'")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"🔍 Search Query: '{query_text}'")
    print(f"🎯 Mode: {mode.upper()}")
    print(f"📊 Top K: {top_k}")
    if mode == "hybrid":
        print(f"⚖️  Weights: Vector={vector_weight}, BM25={bm25_weight}")
    print(f"{'='*80}\n")
    
    # Connect to pgvector database
    print("🔌 Connecting to pgvector database...")
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        dbname="vector_db",
        user="postgres",
        password="postgres"
    )
    cursor = conn.cursor()
    
    # Perform search based on mode
    results = []
    
    if mode == "vector":
        query_embedding = generate_embedding(query_text)
        results = vector_search(cursor, query_embedding, top_k)
        score_label = "Similarity Score"
        
    elif mode == "bm25":
        results = bm25_search(cursor, query_text, top_k)
        score_label = "BM25 Score"
        
    elif mode == "hybrid":
        query_embedding = generate_embedding(query_text)
        
        # Get results from both methods (fetch more for better fusion)
        fetch_k = top_k * 2  # Fetch 2x results for fusion
        vector_results = vector_search(cursor, query_embedding, fetch_k)
        bm25_results = bm25_search(cursor, query_text, fetch_k)
        
        # Combine using RRF
        results = reciprocal_rank_fusion(
            vector_results, 
            bm25_results, 
            vector_weight=vector_weight,
            bm25_weight=bm25_weight
        )[:top_k]  # Take top_k after fusion
        score_label = "RRF Score"
    
    # Display results
    if not results:
        print("❌ No results found. Make sure you have data in the database.")
    else:
        print(f"✅ Found {len(results)} results:\n")
        print("=" * 80)
        
        for i, (doc_id, content, filename, chunk_index, score) in enumerate(results, 1):
            print(f"\n🔢 Result #{i}")
            print(f"📄 File: {filename} (Chunk {chunk_index})")
            print(f"🎯 {score_label}: {score:.4f}")
            print(f"📝 Content Preview: {content[:200]}...")
            print("-" * 80)
    
    cursor.close()
    conn.close()
    print("\n✅ Search complete!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    # Parse arguments
    query = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "hybrid"
    top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    
    similarity_search(query, mode=mode, top_k=top_k)