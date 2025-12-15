import os
import asyncio
import numpy as np
import time
from typing import List, Dict, Any
from dotenv import load_dotenv

# --- Third Party Libraries ---
from groq import AsyncGroq
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
# Get your free key at: https://console.groq.com/keys
GROQ_API_KEY = os.getenv("YOUR_GROQ_KEY_HERE")

class GovernanceEngine:
    """
    The "Guardrail" Layer.
    Uses latent space embeddings (BERT) to mathematically verify 
    if the LLM's answer is supported by the Source Context.
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        print(f" Initializing Governance Engine ({model_name})...")
        # Loads a local BERT model (384 dim)
        self.encoder = SentenceTransformer(model_name)

    def calculate_trust_score(self, answer: str, context: str) -> float:
        """
        Calculates Cosine Similarity between Answer and Context.
        Returns a Trust Score (0-100%).
        """
        if not answer or not context:
            return 0.0

        # 1. Vectorize text
        vec_answer = self.encoder.encode(answer)
        vec_context = self.encoder.encode(context)

        # 2. Compute Cosine Similarity
        dot_product = np.dot(vec_answer, vec_context)
        norm_a = np.linalg.norm(vec_answer)
        norm_b = np.linalg.norm(vec_context)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        similarity = dot_product / (norm_a * norm_b)
        return round(max(0, float(similarity)) * 100, 2)


class PDFIngestion:
    """
    Helper class to handle physical document parsing.
    """
    @staticmethod
    def load_pdf(file_path: str) -> List[str]:
        """
        Reads a PDF and splits it into chunks (simulating pages/paragraphs).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF not found at: {file_path}")
            
        print(f" Parsing PDF: {file_path}...")
        reader = PdfReader(file_path)
        chunks = []
        
        # Strategy: Chunk by Page. 
        # In a real enterprise app, we would use a sliding window of 500 tokens.
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                # Add metadata (Page number) to the text so the LLM knows where it came from
                clean_text = f"[Page {i+1}] {text.strip()}"
                chunks.append(clean_text)
                
        print(f" Extracted {len(chunks)} pages/chunks from PDF.")
        return chunks


class OpenSeekOrchestrator:
    """
    The Main Controller.
    Manages: ChromaDB (Memory) <-> Groq (Brain) <-> Governance (Judge)
    """
    def __init__(self):
        # 1. Setup Inference (Groq)
        if "YOUR_GROQ" in GROQ_API_KEY:
            raise ValueError(" Please set a valid GROQ_API_KEY in the script.")
        self.llm_client = AsyncGroq(api_key=GROQ_API_KEY)

        # 2. Setup Vector DB (Chroma)
        print(" Initializing In-Memory Vector Database...")
        self.db_client = chromadb.Client()
        self.collection = self.db_client.get_or_create_collection(
            name="corporate_knowledge",
            metadata={"hnsw:space": "cosine"}
        )

        # 3. Setup Governance
        self.governance = GovernanceEngine()

    def ingest_documents(self, documents: List[str]):
        """Load text chunks into the Vector Store."""
        print(f" Indexing {len(documents)} document chunks...")
        ids = [f"id_{time.time()}_{i}" for i in range(len(documents))]
        self.collection.add(documents=documents, ids=ids)

    async def _generate_answer(self, query: str, context: str) -> str:
        """Call the LLM (Llama-3) to draft an answer."""
        try:
            chat = await self.llm_client.chat.completions.create(
                messages=[
                    {
                        "role": "system", 
                        "content": f"You are a helpful assistant. Answer the user question using ONLY this context:\n\n{context}"
                    },
                    {"role": "user", "content": query}
                ],
                model="llama3-8b-8192",
                temperature=0.1 # Keep it factual
            )
            return chat.choices[0].message.content
        except Exception as e:
            return f"LLM Error: {str(e)}"

    async def process_query(self, user_query: str):
        """
        The Full Governed Pipeline.
        """
        print(f"\n User Query: '{user_query}'")
        
        # 1. Retrieval (RAG)
        results = self.collection.query(query_texts=[user_query], n_results=1)
        
        if not results['documents'][0]:
            return {"status": "ERROR", "msg": "No data found."}
            
        retrieved_context = results['documents'][0][0]
        
        # 2. Generation (LLM)
        answer = await self._generate_answer(user_query, retrieved_context)

        # 3. Governance (The Guardrail)
        score = self.governance.calculate_trust_score(answer, retrieved_context)
        
        # 4. Gatekeeper Logic
        status = "APPROVED" if score >= 50.0 else "BLOCKED"
        
        return {
            "query": user_query,
            "status": status,
            "trust_score": score,
            "answer": answer,
            "context_snippet": retrieved_context[:100] + "..."
        }

# --- MAIN EXECUTION ---
async def main():
    system = OpenSeekOrchestrator()

    # --- Step 1: Ingest Data ---
    # OPTION A: Load from a PDF (Uncomment this if you have a real file)
    # pdf_chunks = PDFIngestion.load_pdf("my_academic_paper.pdf")
    # system.ingest_documents(pdf_chunks)

    # OPTION B: Dummy Data (For testing)
    print("\n--- 1. Ingesting Data ---")
    dummy_data = [
        "Refund Policy: Customers are eligible for a full refund within 30 days of purchase if the item is unused. Return shipping is free.",
        "Security Policy: All employees must use 2FA. Passwords rotate every 90 days.",
        "Remote Work: Remote work is allowed for up to 2 days per week for engineering teams."
    ]
    system.ingest_documents(dummy_data)

    print("\n" + "="*50)
    print(" OPENSEEK SYSTEM ONLINE")
    print("="*50)

    # --- Step 2: Test the System ---
    
    # Test Case 1: Valid Question
    result1 = await system.process_query("What is the refund deadline?")
    print(f"\n SCENARIO 1 (Valid):")
    print(f"   Status:      {result1['status']}")
    print(f"   Trust Score: {result1['trust_score']}%")
    print(f"   Answer:      {result1['answer']}")

    # Test Case 2: Hallucination Attempt
    # (Asking about something not in the policy)
    result2 = await system.process_query("Can I get a refund after 6 months?")
    print(f"\n SCENARIO 2 (Invalid/Hallucination Check):")
    print(f"   Status:      {result2['status']}")
    print(f"   Trust Score: {result2['trust_score']}%")
    if result2['status'] == "BLOCKED":
        print(f"   Action: Answer suppressed by Governance Layer.")
    else:
        print(f"   Answer: {result2['answer']}")

if __name__ == "__main__":
    asyncio.run(main())
