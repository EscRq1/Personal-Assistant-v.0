import requests
import datetime
import socket
import os
import numpy as np
from tools.web_search import search_web, fetch_page_text, is_online

class Controller:
    def __init__(self, model_name="qwen3:7b"):      
        self.router_model = "qwen2.5:0.5b"
        self.researcher_model = "qwen2.5:7b"     # Generalist
        self.coder_model = "qwen3-coder:30b"    # Coding specialist
        self.url = "http://localhost:11434/api/generate"
        self.last_intent = None
        
        self.temps = {   
                "ROUTER": 0.0,
                "RESEARCHER": 0.8,  # 0.8 = More "wit" and conversational flow
                "CODER": 0.2        # 0.2 = Strict syntax, less "hallucinated" code
        }

        self.context_size = {
            "ROUTER": 2048,
            "RESEARCHER": 8192,
            "CODER": 16384
        }




    def _call_ollama(self, model, prompt, system_prompt=None, target_temp = 0, target_context = 8192):
        # Concatenate for /api/generate reliability
        if system_prompt:
            final_prompt = f"### SYSTEM INSTRUCTIONS ###\n{system_prompt}\n\n### USER INPUT ###\n{prompt}"
        else:
            final_prompt = prompt

        payload = {
            "model": model,
            "prompt": final_prompt,
            "stream": False,
            "options": {
                "temperature": target_temp,
                "num_ctx": target_context,
                "low_vram": True
            }
        }
        
        try:
            response = requests.post(self.url, json=payload)
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            return f"Error: {e}"


    def ask(self, prompt: str, system_prompt=None, intent="RESEARCH") -> str:
        """Routes to the specialist model based on intent."""
        # Selection Logic
        if "CODE" in intent:
            model = self.coder_model
            temp = self.temps["CODER"]
            context = self.context_size["CODER"]
        else:
            model = self.researcher_model
            temp = self.temps["RESEARCHER"]
            context = self.context_size["RESEARCHER"]

        return self._call_ollama(model, prompt, system_prompt, target_temp = temp, target_context = context)

    def generate_search_query(self, history: list, user_input: str) -> str:
        """
        Uses conversation history to turn 'search for it' into 'Java 25 features'
        """
        # Create a text block of the last 3 turns
        context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-3:]])
        
        prompt = (
            f"Conversation History:\n{context}\n\n"
            f"User's new command: '{user_input}'\n\n"
            "Task: The user wants to search the web. Based on the history and command, "
            "write the best single search query to put into a search engine. "
            "Output ONLY the search query, nothing else."
        )
        
        query = self.ask(prompt).strip().strip('"').strip("'")
        print(f"DEBUG: Generated Search Query -> {query}") # Helpful for debugging
        return query

    def web_lookup(self, query: str) -> str:
    # 1. Check if we even have internet before trying
        if not is_online():
            print(" [Offline] Skipping web search... using local brain.")
            return ""

        print(f"DEBUG: Searching web for -> {query}")
        context_chunks = []
    
        try:
        # 2. Perform the search
            search_results = search_web(query, max_results=3)
        
            for r in search_results:
                url = r.get("href") or r.get("url")
                if not url: continue
            
            # 3. Fetch the content
                text = fetch_page_text(url)
                if text:
                    context_chunks.append(f"Source ({url}):\n{text}")
                
        except Exception as e:
        # Catching DDGSException or Timeout errors here
            print(f" [Error] Web search failed: {e}")
            return ""

        return "\n\n".join(context_chunks)



# No 'import ollama' needed!

class SemanticController(Controller):
    def __init__(self):
        super().__init__()
        # 1. Configuration
        self.embed_url = "http://localhost:11434/api/embed"
        self.embed_model = "nomic-embed-text"
        self.cache_file = "intent_anchors.npy"
        
        # Prototype examples for each intent
        self.raw_anchors = {
            "CODE": [
                # Creation
                "write", "implement", "create", "build",
                "generate", "code", "program", "develop", "script",
                "debug", "fix"
                # Modification (Crucial for follow-ups!)
                "make it faster", "refactor this", 
                "same task",
                "same thing",
                "same but",
                "what about",
                "now in",
                "but in",
                # Debugging
                "debug error", "fix the bug", "why is it crashing"
            ],
            "RESEARCH": [
                # Knowledge
                "explain concept", "how does this work", "tell me about", "find some information", "search for it"
                # Conversational / Feedback (The missing link)
                "thank you",
                "thanks",
                "that helps",
                "that really helps",
                "nice",
                "cool",
                "got it",
                "makes sense",
                "i see",
                "interesting"
            ]
        }

        # 2. Boot Optimization (Persistence)
        if os.path.exists(self.cache_file):
            print(" [System] Loading semantic anchors from cache...")
            self.anchor_vectors = np.load(self.cache_file, allow_pickle=True).item()
        else:
            print(" [System] No cache found. Generating new anchors...")
            self._generate_and_save_anchors()
        
        self.last_intent = "RESEARCH" # Default context

    def _get_embeddings(self, input_data):
        """Standardized wrapper for Ollama's /api/embed endpoint."""
        # Ollama expects 'input' to be a string or a list of strings
        payload = {"model": self.embed_model, "input": input_data}
        try:
            response = requests.post(self.embed_url, json=payload, timeout=10)
            response.raise_for_status()
            vectors = response.json().get("embeddings")
            
            if not vectors:
                print(f" [!] API returned no embeddings for: {input_data[:20]}...")
                return None
            return vectors
        except Exception as e:
            print(f" [!] Embedding Error: {e}")
            return None

    def _generate_and_save_anchors(self):
        self.anchor_vectors = {}
        for intent, phrases in self.raw_anchors.items():
            print(f" -> Creating centroid for {intent}...")
            vectors = self._get_embeddings(phrases)
            
            if vectors:
                # Calculate the "Centroid" (Average vector for this category)
                centroid = np.mean(vectors, axis=0)
                # Validation: Magnitude check
                mag = np.linalg.norm(centroid)
                if mag == 0:
                    print(f" [!!!] Warning: {intent} centroid is a zero vector!")
                
                self.anchor_vectors[intent] = centroid
            else:
                self.anchor_vectors[intent] = np.zeros(768)
        
        np.save(self.cache_file, self.anchor_vectors)
        print(f" [Success] Cached anchors to {self.cache_file}")

    def _cosine_similarity(self, a, b):
        """Calculates distance in vector space."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)

    def get_intent(self, user_input) -> str:
        # Get vector for user input (returns a list of lists, we take the first)
        result = self._get_embeddings(user_input)
        if not result:
            return self.last_intent
            
        input_vec = result[0]
        
        # Calculate similarity scores against our anchor centroids
        scores = {intent: self._cosine_similarity(input_vec, anchor) 
                  for intent, anchor in self.anchor_vectors.items()}
        
        # Find winner
        best_intent = max(scores, key=scores.get)
        confidence = scores[best_intent]

        # --- Threshold Logic ---
        # If the confidence is too low, we stick to the last intent 
        # (This handles follow-ups like "make it better" or "thanks")
        if confidence < 0.45:
            print(f" [Router] Ambiguous ({confidence:.2f}). Defaulting to: {self.last_intent}")
            return self.last_intent

        self.last_intent = best_intent
        print(f" [Router] Selected: {best_intent} ({confidence:.2f})")
        return best_intent