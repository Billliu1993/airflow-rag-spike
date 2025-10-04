"""
Tasks for extracting and converting documents to markdown.
"""

import logging
import os
from pathlib import Path
from typing import List, Dict

import psycopg2
from markitdown import MarkItDown
from openai import OpenAI

logger = logging.getLogger(__name__)


def get_processed_filepaths(
    db_host: str = "localhost",
    db_port: int = 5432,
    db_name: str = "vector_db",
    db_user: str = "postgres",
    db_password: str = "postgres",
    table_name: str = "documents",
) -> set[str]:
    """
    Get list of filepaths that have already been processed (exist in metadata).
    
    Args:
        db_host: Database host
        db_port: Database port
        db_name: Database name
        db_user: Database user
        db_password: Database password
        table_name: Name of the table to check
        
    Returns:
        Set of processed filepaths
    """
    conn = None
    processed_files = set()
    
    try:
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            dbname=db_name,
            user=db_user,
            password=db_password,
        )
        cursor = conn.cursor()
        
        # Get all filepaths from metadata
        cursor.execute(f"""
            SELECT DISTINCT metadata->>'filepath' as filepath
            FROM {table_name}
            WHERE metadata ? 'filepath'
            AND metadata->>'filepath' IS NOT NULL
        """)
        
        results = cursor.fetchall()
        processed_files = {row[0] for row in results if row[0]}
        
        logger.info(f"Found {len(processed_files)} already processed files in database")
        cursor.close()
        
    except Exception as e:
        logger.error(f"Error getting processed filepaths: {str(e)}")
        raise
    finally:
        if conn:
            conn.close()
    
    return processed_files


def extract_documents(
    raw_documents_path: str = "/Users/bill/Projects/airflow-rag-spike/include/raw_documents",
    db_host: str = "localhost",
    db_port: int = 5432,
    db_name: str = "vector_db",
    db_user: str = "postgres",
    db_password: str = "postgres",
    table_name: str = "documents",
    use_llm: bool = True,
    llm_model: str = "gpt-4o-mini",
    llm_api_key: str = None,
) -> List[Dict[str, str]]:
    """
    Extract and convert documents from raw_documents folder to markdown.
    
    Only processes files that haven't been processed yet (not in metadata).
    
    Args:
        raw_documents_path: Path to raw documents folder
        db_host: Database host
        db_port: Database port
        db_name: Database name
        db_user: Database user
        db_password: Database password
        table_name: Name of the table to check for processed files
        use_llm: Whether to use LLM for enhanced extraction (default: True)
        llm_model: OpenAI model to use for extraction (default: gpt-4o-mini)
        llm_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        
    Returns:
        List of extracted documents with their metadata
    """
    logger.info(f"Starting document extraction from {raw_documents_path}")
    
    # Get already processed files
    processed_files = get_processed_filepaths(
        db_host=db_host,
        db_port=db_port,
        db_name=db_name,
        db_user=db_user,
        db_password=db_password,
        table_name=table_name,
    )
    
    # Initialize MarkItDown with optional LLM support
    if use_llm:
        # Get API key
        if llm_api_key is None:
            llm_api_key = os.environ.get("OPENAI_API_KEY")
        
        if llm_api_key:
            logger.info(f"Initializing MarkItDown with LLM support (model: {llm_model})")
            # Create OpenAI client for MarkItDown with financial domain instructions
            llm_client = OpenAI(api_key=llm_api_key)
            
            # Custom prompt for financial document extraction
#             system_prompt = """You are extracting content from financial documents. Follow these guidelines:

# 1. TABLES: Preserve all tables in markdown format with proper alignment. Include all headers and values exactly as shown.
# 2. NUMBERS: Maintain exact numerical values, percentages, and financial figures.
# 3. STRUCTURE: Keep the document structure with proper headings (use # ## ### for hierarchy).
# 4. DATES: Preserve all dates in their original format.
# 5. FORMATTING: Bold important terms, use bullet points for lists.
# 6. CONTEXT: Keep financial terms, metrics, and KPIs with their full context.
# 7. ACCURACY: Never summarize or paraphrase - extract verbatim.

# Output clean, well-structured markdown that preserves all information."""
            
            md_converter = MarkItDown(
                llm_client=llm_client, 
                llm_model=llm_model,
                # llm_prompt=system_prompt
            )
        else:
            logger.warning("LLM requested but OPENAI_API_KEY not found. Falling back to basic extraction.")
            md_converter = MarkItDown()
    else:
        logger.info("Using basic MarkItDown extraction (no LLM)")
        md_converter = MarkItDown()
    
    # Get all files from raw_documents folder
    raw_docs_dir = Path(raw_documents_path)
    if not raw_docs_dir.exists():
        logger.warning(f"Raw documents directory does not exist: {raw_documents_path}")
        return []
    
    all_files = [f for f in raw_docs_dir.iterdir() if f.is_file()]
    logger.info(f"Found {len(all_files)} total files in {raw_documents_path}")
    
    # Filter out already processed files
    files_to_process = [
        f for f in all_files 
        if str(f.absolute()) not in processed_files
    ]
    
    if not files_to_process:
        logger.info("No new files to process. All files have been processed already.")
        return []
    
    logger.info(f"Processing {len(files_to_process)} new files...")
    
    extracted_docs = []
    
    for file_path in files_to_process:
        try:
            logger.info(f"Converting {file_path.name} to markdown...")
            
            # Convert file to markdown using MarkItDown
            result = md_converter.convert(str(file_path))
            markdown_content = result.text_content
            
            # Create document with metadata
            doc = {
                "filepath": str(file_path.absolute()),
                "filename": file_path.name,
                "content": markdown_content,
                "file_size": file_path.stat().st_size,
                "file_extension": file_path.suffix,
            }
            
            extracted_docs.append(doc)
            logger.info(f"Successfully converted {file_path.name} ({len(markdown_content)} characters)")
            
        except Exception as e:
            logger.error(f"Error processing file {file_path.name}: {str(e)}")
            continue
    
    logger.info(f"Successfully extracted {len(extracted_docs)} documents")
    
    return extracted_docs


