# OpenSeek
A Governed Multi-Agent RAG System
Author: Yash Gopani
Date: 2025-09-14
Description: 
    A local implementation of an Enterprise AI platform. 
    It features a custom Orchestration Layer, Vector Retrieval (RAG), 
    and a Deterministic Governance Layer using Cosine Similarity to prevent hallucinations.

Architecture:
    1. Retrieval: ChromaDB (Vector Store)
    2. Inference: Groq LPU (Llama-3-8b)
    3. Governance: Sentence-Transformers (all-MiniLM-L6-v2)
