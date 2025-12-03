# chatbot.py
"""
Interactive Harry Potter RAG Chatbot
"""

import os
import google.generativeai as genai
from src.embedding import load_model
from src.similarity_search import search_similar_chunks
from typing import List, Tuple
import pickle
from sentence_transformers import CrossEncoder
import numpy as np
from rank_bm25 import BM25Okapi
import faiss

os.environ['GEMINI_API_KEY'] = "AIzaSyBER6P2JhStWk5rrcVAiSm_oIrjNM5Wuw0" 

class HarryPotterChatbot:
    """
    Interactive chatbot for Harry Potter books.
    """
    
    def __init__(self): # Removed embeddings_path arg as we hardcode the faiss path
        """Initialize the chatbot with FAISS."""
        print("Initializing Harry Potter RAG System...")
        
        # 1. Load Models
        self.model = load_model()
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # 2. Load Text Chunks (We still need the text!)
        print("Loading text chunks...")
        with open("data/chunks.pkl", 'rb') as f:
            self.chunks = pickle.load(f)

        # 3. Load FAISS Index (Instead of embeddings.pkl)
        print("Loading FAISS Vector DB...")
        self.faiss_index = faiss.read_index("data/vector_store.index")
        
        # 4. Build BM25
        print("Building keyword search index...")
        chunk_texts = [chunk.page_content for chunk in self.chunks]
        tokenized_corpus = [doc.split(" ") for doc in chunk_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # 5. Gemini
        api_key = os.environ.get("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        
        self.conversation_history = []
        print("System Ready!")

    
    def create_prompt(self, question: str, relevant_chunks: List[Tuple[int, float]], 
                      include_history: bool = False) -> str:
        """Create prompt with context and optional conversation history."""
        
        # Build context from chunks
        context_parts = []
        for rank, (idx, score) in enumerate(relevant_chunks, 1):
            chunk = self.chunks[idx]
            source = chunk.metadata.get('source', 'Unknown')
            chapter = chunk.metadata.get('chapter', 'Unknown')
            text = chunk.page_content
            context_parts.append(
                f"[Passage {rank} - {source}, Chapter {chapter}]\n{text}"
            )
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Add conversation history if requested
        history_text = ""
        if include_history and self.conversation_history:
            history_text = "\n\nPREVIOUS CONVERSATION:\n"
            for prev_q, prev_a in self.conversation_history[-3:]:  # Last 3 exchanges
                history_text += f"Q: {prev_q}\nA: {prev_a[:200]}...\n\n"
        
        # Create prompt
        prompt = f"""You are a helpful assistant answering questions about Harry Potter books. 

I will provide you with relevant passages from the books, and you should answer the question based on these passages.
{history_text}
RETRIEVED PASSAGES:
{context}

QUESTION: {question}

Please provide a clear and accurate answer based on the passages above. If the passages don't contain enough information, say so. Always cite which book and chapter your information comes from."""
        
        return prompt
    
    def query(self, question: str, top_k: int = 10, verbose: bool = True) -> dict:
        if verbose: print(f"\nSearching for: '{question}'")
        
        # --- Step 1: Hybrid Search ---
        
        # A. FAISS Vector Search
        query_vector = self.model.encode([question]).astype('float32')
        retrieval_k = 25
        
        # Search index returns (distances, indices)
        distances, faiss_indices = self.faiss_index.search(query_vector, retrieval_k)
        # faiss_indices is a list of lists (for batches), we want the first list
        vector_indices = set(faiss_indices[0])

        # B. BM25 Keyword Search
        tokenized_query = question.split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_indices = set(np.argsort(bm25_scores)[::-1][:retrieval_k])
        
        # Combine
        combined_indices = vector_indices.union(bm25_indices)
        
        # --- Step 2: Rerank (Same as before) ---
        rerank_data = [(idx, self.chunks[idx].page_content) for idx in combined_indices if idx != -1]
        rerank_pairs = [(question, content) for idx, content in rerank_data]
        
        if not rerank_pairs:
             return {'answer': "I couldn't find any relevant information."}

        reranker_scores = self.reranker.predict(rerank_pairs)
        
        reranked_results = []
        for i, score in enumerate(reranker_scores):
            idx = rerank_data[i][0]
            reranked_results.append((idx, score))
        
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        results = reranked_results[:top_k]
        
        # Step 3: Expand context
        indices_to_include = set()
        for idx, score in results:
            indices_to_include.add(idx)
            if idx > 0:
                indices_to_include.add(idx - 1)
            if idx < len(self.chunks) - 1:
                indices_to_include.add(idx + 1)
        
        final_chunks = [(idx, 0.0) for idx in sorted(indices_to_include)]
        
        # Step 4: Create prompt and query Gemini
        prompt = self.create_prompt(question, final_chunks, include_history=True)
        
        if verbose:
            print("Thinking...")
        
        try:
            response = self.gemini_model.generate_content(prompt)
            answer = response.text
        except Exception as e:
            answer = f"Error: {e}"
        
        return {
            'question': question,
            'answer': answer,
            'retrieved_chunks': results
        }
    
    def display_answer(self, answer: str):
        """Pretty print the answer."""
        print(f"\n{'='*80}")
        print("ANSWER:")
        print(f"{'='*80}\n")
        print(answer)
        print(f"\n{'='*80}\n")
    
    def show_history(self):
        """Display conversation history."""
        if not self.conversation_history:
            print("\n No conversation history yet.\n")
            return
        
        print(f"\n{'='*80}")
        print("CONVERSATION HISTORY")
        print(f"{'='*80}\n")
        
        for i, (q, a) in enumerate(self.conversation_history, 1):
            print(f"{i}. Q: {q}")
            print(f"   A: {a[:150]}...\n")
        
        print(f"{'='*80}\n")
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        print("\n Conversation history cleared.\n")
    
    def chat(self):
        """Start interactive chat loop."""
        
        while True:
            try:
                # Get user input
                question = input("You: ").strip()
                
                # Check for exit commands
                if question.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("\n Goodbye! Thanks for chatting about Harry Potter!\n")
                    break
                
                # Check for special commands
                if question.lower() == 'history':
                    self.show_history()
                    continue
                
                if question.lower() == 'clear':
                    self.clear_history()
                    continue
                
                # Empty input
                if not question:
                    print("Please ask a question!\n")
                    continue
                
                # Query the system
                result = self.query(question, top_k=10, verbose=True)
                
                # Display answer
                self.display_answer(result['answer'])
                
                # Save to history
                self.conversation_history.append((question, result['answer']))
                
            except KeyboardInterrupt:
                print("\n\n Goodbye!\n")
                break
            except Exception as e:
                print(f"\n Error: {e}\n")


def main():
    """Main function to run the chatbot."""
    
    # Initialize chatbot
    chatbot = HarryPotterChatbot(
        embeddings_path="data/embeddings.pkl"
    )
    
    # Start chatting
    chatbot.chat()


if __name__ == "__main__":
    main()