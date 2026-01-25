-- Khora PostgreSQL Initialization
--
-- This script runs only on first container initialization (new volume).
-- It enables the pgvector extension for vector similarity search.

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
