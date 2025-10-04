"""
Tasks for managing vector database tables in pgvector.
"""

import logging
from typing import Optional

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

logger = logging.getLogger(__name__)


def get_vector_dimension(cursor, table_name: str) -> Optional[int]:
    """
    Get the vector dimension of an existing table.
    
    Args:
        cursor: Database cursor
        table_name: Name of the table to check
        
    Returns:
        The vector dimension if table exists, None otherwise
    """
    # Check if table exists and get vector column dimension
    cursor.execute("""
        SELECT a.atttypmod
        FROM pg_attribute a
        JOIN pg_class c ON a.attrelid = c.oid
        JOIN pg_type t ON a.atttypid = t.oid
        WHERE c.relname = %s
        AND t.typname = 'vector'
        AND a.attname = 'embedding'
    """, (table_name,))
    
    result = cursor.fetchone()
    if result and result[0] > 0:
        return result[0]
    return None


def create_vector_table(
    table_name: str = "documents",
    vector_dim: int = 512,
    db_host: str = "localhost",
    db_port: int = 5432,
    db_name: str = "vector_db",
    db_user: str = "postgres",
    db_password: str = "postgres",
) -> dict:
    """
    Create or manage a vector table in pgvector database.
    
    This task implements the following logic:
    - If table doesn't exist: create it with specified vector dimension
    - If table exists with correct dimension: do nothing
    - If table exists with wrong dimension: drop and recreate
    
    Args:
        table_name: Name of the vector table (default: "documents")
        vector_dim: Dimension of the embedding vectors (default: 768 for FinBERT)
        db_host: Database host (default: "localhost")
        db_port: Database port (default: 5432)
        db_name: Database name (default: "vector_db")
        db_user: Database user (default: "postgres")
        db_password: Database password (default: "postgres")
        
    Returns:
        Dictionary with action taken and table info
    """
    conn = None
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
        
        # Check if table exists and get current vector dimension
        logger.info(f"Checking if table '{table_name}' exists...")
        current_dim = get_vector_dimension(cursor, table_name)
        
        action_taken = None
        
        if current_dim is None:
            # Table doesn't exist, create it
            logger.info(f"Table '{table_name}' does not exist. Creating with vector dimension {vector_dim}...")
            cursor.execute(f"""
                CREATE TABLE {table_name} (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding vector({vector_dim}),
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create vector similarity index (IVFFlat for fast cosine similarity)
            cursor.execute(f"""
                CREATE INDEX {table_name}_vector_idx ON {table_name} 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)
            
            # Create BM25 index for full-text search (hybrid retrieval)
            cursor.execute(f"""
                CREATE INDEX {table_name}_bm25_idx ON {table_name}
                USING bm25 (id, content)
                WITH (key_field='id')
            """)
            
            action_taken = "created"
            logger.info(f"Successfully created table '{table_name}' with vector dimension {vector_dim}")
            logger.info("Created indexes: vector similarity (IVFFlat) + BM25 full-text search")
            
        elif current_dim == vector_dim:
            # Table exists with correct dimension
            logger.info(f"Table '{table_name}' already exists with correct vector dimension {vector_dim}. No action needed.")
            action_taken = "no_change"
            
        else:
            # Table exists but with wrong dimension, drop and recreate
            logger.warning(
                f"Table '{table_name}' exists with vector dimension {current_dim} "
                f"but expected {vector_dim}. Dropping and recreating..."
            )
            cursor.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")
            
            cursor.execute(f"""
                CREATE TABLE {table_name} (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding vector({vector_dim}),
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create vector similarity index
            cursor.execute(f"""
                CREATE INDEX {table_name}_vector_idx ON {table_name} 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)
            
            # Create BM25 index for full-text search
            cursor.execute(f"""
                CREATE INDEX {table_name}_bm25_idx ON {table_name}
                USING bm25 (id, content)
                WITH (key_field='id')
            """)
            
            action_taken = "recreated"
            logger.info(
                f"Successfully recreated table '{table_name}' with vector dimension {vector_dim} "
                f"(was {current_dim})"
            )
            logger.info("Created indexes: vector similarity (IVFFlat) + BM25 full-text search")
        
        cursor.close()
        
        return {
            "table_name": table_name,
            "vector_dimension": vector_dim,
            "action": action_taken,
            "status": "success",
        }
        
    except Exception as e:
        logger.error(f"Error managing vector table: {str(e)}")
        raise
        
    finally:
        if conn:
            conn.close()


