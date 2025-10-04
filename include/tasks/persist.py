"""
Tasks for persisting embeddings and document chunks to vector database.
"""

import logging
from typing import List, Dict
import json

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from psycopg2.extras import execute_batch

logger = logging.getLogger(__name__)


def store_embeddings(
    embedded_chunks: List[Dict[str, any]],
    db_host: str = "localhost",
    db_port: int = 5432,
    db_name: str = "vector_db",
    db_user: str = "postgres",
    db_password: str = "postgres",
    table_name: str = "documents",
    batch_size: int = 100,
) -> Dict[str, int]:
    """
    Store embedded document chunks in pgvector database.
    
    Args:
        embedded_chunks: List of chunks with embeddings from the embedding task
        db_host: Database host
        db_port: Database port
        db_name: Database name
        db_user: Database user
        db_password: Database password
        table_name: Name of the table to insert into
        batch_size: Number of records to insert per batch
        
    Returns:
        Dictionary with statistics about the operation
    """
    if not embedded_chunks:
        logger.info("No embedded chunks to store. Skipping.")
        return {"inserted": 0, "skipped": 0, "failed": 0}
    
    logger.info(f"Starting to store {len(embedded_chunks)} embedded chunks...")
    
    conn = None
    inserted_count = 0
    skipped_count = 0
    failed_count = 0
    
    try:
        # Connect to the database
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            dbname=db_name,
            user=db_user,
            password=db_password,
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Prepare data for insertion
        records_to_insert = []
        
        for chunk in embedded_chunks:
            embedding = chunk.get("embedding")
            content = chunk.get("content")
            metadata = chunk.get("metadata", {})
            
            # Skip chunks without embeddings
            if embedding is None:
                logger.warning(f"Skipping chunk without embedding: {metadata.get('filename', 'unknown')}")
                skipped_count += 1
                continue
            
            # Skip chunks without content
            if not content:
                logger.warning(f"Skipping chunk without content: {metadata.get('filename', 'unknown')}")
                skipped_count += 1
                continue
            
            records_to_insert.append((
                content,
                embedding,
                json.dumps(metadata)
            ))
        
        if not records_to_insert:
            logger.warning("No valid records to insert after filtering.")
            return {"inserted": 0, "skipped": skipped_count, "failed": 0}
        
        logger.info(f"Inserting {len(records_to_insert)} records into {table_name}...")
        
        # Insert in batches
        insert_query = f"""
            INSERT INTO {table_name} (content, embedding, metadata)
            VALUES (%s, %s::vector, %s::jsonb)
        """
        
        try:
            execute_batch(cursor, insert_query, records_to_insert, page_size=batch_size)
            inserted_count = len(records_to_insert)
            logger.info(f"Successfully inserted {inserted_count} records")
        except Exception as e:
            logger.error(f"Error during batch insert: {str(e)}")
            failed_count = len(records_to_insert)
            raise
        
        cursor.close()
        
    except Exception as e:
        logger.error(f"Error storing embeddings: {str(e)}")
        raise
    finally:
        if conn:
            conn.close()
    
    result = {
        "inserted": inserted_count,
        "skipped": skipped_count,
        "failed": failed_count,
        "total": len(embedded_chunks),
    }
    
    logger.info(f"Storage complete: {result}")
    return result

