# pgvector + BM25 Database

PostgreSQL with pgvector extension for vector similarity search and pg_search extension for BM25 full-text search.

## Start the database

```bash
cd pg-vector
docker-compose up -d
```

## Stop the database

```bash
cd pg-vector
docker-compose down
```

## Remove the database (including data)

```bash
cd pg-vector
docker-compose down -v
```

## Connection Details

- **Host:** `localhost`
- **Port:** `5432`
- **Database:** `vector_db`
- **User:** `postgres`
- **Password:** `postgres`

## Quick Test

Verify vector search is working:

```bash
# Connect to the database
docker exec -it pgvector_db psql -U postgres -d vector_db

# Inside psql, run these commands:
```

```sql
-- Create a test table
CREATE TABLE test_embeddings (id serial PRIMARY KEY, embedding vector(3));

-- Insert test vectors
INSERT INTO test_embeddings (embedding) VALUES 
  ('[1,2,3]'),
  ('[4,5,6]'),
  ('[7,8,9]');

-- Test vector similarity search (cosine distance)
SELECT id, embedding, embedding <-> '[3,3,3]' AS distance 
FROM test_embeddings 
ORDER BY distance 
LIMIT 3;

-- Clean up
DROP TABLE test_embeddings;

-- Exit psql
\q
```

### Test BM25 Search

```sql
-- Create a test table for BM25
CREATE TABLE test_documents (
  id SERIAL PRIMARY KEY,
  title TEXT,
  body TEXT
);

-- Insert test documents
INSERT INTO test_documents (title, body) VALUES
  ('Machine Learning Basics', 'An introduction to machine learning and neural networks'),
  ('Deep Learning Advanced', 'Advanced concepts in deep learning and AI'),
  ('Python Programming', 'Learn Python programming from scratch');

-- Create a BM25 index (key_field must be listed first)
CREATE INDEX test_documents_idx ON test_documents
USING bm25 (id, title, body)
WITH (key_field='id');

-- Test BM25 search using the @@@ operator
SELECT title, body, paradedb.score(id) as score
FROM test_documents
WHERE title @@@ 'machine' OR body @@@ 'learning'
ORDER BY score DESC
LIMIT 5;

-- Test phrase search
SELECT title, body
FROM test_documents
WHERE body @@@ '"machine learning"'
LIMIT 5;

-- Clean up
DROP TABLE test_documents CASCADE;
```

