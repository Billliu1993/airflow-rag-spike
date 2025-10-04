"""
RAG Ingestion Pipeline DAG

This DAG manages the ingestion of documents into a vector database for RAG (Retrieval Augmented Generation).
"""

from datetime import datetime

from airflow.sdk import dag, task

from include.tasks.vector_db import create_vector_table
from include.tasks.extract import extract_documents
from include.tasks.chunk import chunk_documents
from include.tasks.embed import generate_embeddings
from include.tasks.persist import store_embeddings


@dag(
    dag_id="rag_ingestion_pipeline",
    description="RAG document ingestion pipeline with vector database",
    schedule=None,  # Manual trigger only - run when you add new documents
    start_date=datetime(2025, 10, 1),
    catchup=False,
    tags=["rag", "ingestion", "vector"],
    default_args={
        "owner": "airflow",
        "retries": 1,
    },
)
def rag_ingestion_pipeline():
    """
    RAG ingestion pipeline that sets up vector database and processes documents.
    """
    
    @task(task_id="setup_vector_table")
    def setup_vector_table_task():
        """Setup vector table in pgvector database."""
        return create_vector_table(
            table_name="documents",
            vector_dim=512,  # OpenAI text-embedding-3-small dimension
            db_host="localhost",
            db_port=5432,
            db_name="vector_db",
            db_user="postgres",
            db_password="postgres",
        )
    
    @task(task_id="extract_documents")
    def extract_documents_task():
        """Extract and convert raw documents to markdown using LLM."""
        return extract_documents(
            raw_documents_path="/Users/bill/Projects/airflow-rag-spike/include/raw_documents",
            db_host="localhost",
            db_port=5432,
            db_name="vector_db",
            db_user="postgres",
            db_password="postgres",
            table_name="documents",
            use_llm=True,
            llm_model="gpt-4o-mini",
        )
    
    @task(task_id="chunk_documents")
    def chunk_documents_task(extracted_docs):
        """Chunk extracted documents into smaller pieces."""
        return chunk_documents(
            extracted_docs=extracted_docs,
            chunk_size=500,
            chunk_overlap=50,
        )
    
    @task(task_id="generate_embeddings")
    def generate_embeddings_task(chunks):
        """Generate embeddings for document chunks using OpenAI."""
        return generate_embeddings(
            chunks=chunks,
            model_name="text-embedding-3-small",
            embedding_dimensions=512,
        )
    
    @task(task_id="store_embeddings")
    def store_embeddings_task(embedded_chunks):
        """Store embedded chunks in pgvector database."""
        return store_embeddings(
            embedded_chunks=embedded_chunks,
            db_host="localhost",
            db_port=5432,
            db_name="vector_db",
            db_user="postgres",
            db_password="postgres",
            table_name="documents",
            batch_size=100,
        )
    
    # Execute tasks and define dependencies
    vector_setup = setup_vector_table_task()
    extracted_docs = extract_documents_task()
    chunks = chunk_documents_task(extracted_docs)
    embeddings = generate_embeddings_task(chunks)
    storage_result = store_embeddings_task(embeddings)
    
    # Define task dependencies
    vector_setup >> extracted_docs >> chunks >> embeddings >> storage_result


# Instantiate the DAG
rag_ingestion_pipeline()