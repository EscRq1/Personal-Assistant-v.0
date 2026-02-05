import requests
import datetime
import socket
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

        
  

    def is_action_feedback(self, user_input) -> bool:
        feedback_phrases = {
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
        }
        text_l = user_input.lower()

        return any(p in text_l for p in feedback_phrases)

    

    def is_obvious_code_request(self, text: str) -> bool:
        code_action_verbs = {
            "write", "implement", "create", "build",
            "generate", "code", "program", "develop", "script",
            "debug", "fix"
        }
        code_domain_terms = {
            "golang", "go", "python", "c++", "java", "rust",
            "javascript", "js", "typescript", "ts",
            "electron", "react", "node", "api", "class", "function",
            "calculator", "gui", "frontend", "backend"

        }
        text_l = text.lower()

        has_action = any(v in text_l for v in code_action_verbs)
        has_domain = any(d in text_l for d in code_domain_terms)

        def starts_with_verb(text: str) -> bool:
            return text.strip().split()[0].lower() in code_action_verbs

        return has_domain and (has_action or starts_with_verb(text))

    def is_code_followup(self, text: str) -> bool:

        followup_code_phrases = {
            "same task",
            "same thing",
            "same but",
            "what about",
            "now in",
            "but in",
            "in rust",
            "in python",
            "in go",
            "in java",
            "in js",
            "in cpp",
        }
        text_l = text.lower()
        return any(p in text_l for p in followup_code_phrases)



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

    def get_intent(self, user_input) -> str:
        rule_intent = None
        text_l = user_input.lower()

        # Use temperature 0.1 for strict, fast classification
        if self.is_obvious_code_request(user_input):
            intent = "CODE"
            last_intent = "CODE"
            return intent
        elif self.last_intent  == "CODE" and self.is_code_followup(user_input) :
            rule_intent = "CODE"   # hint only, not a decision
        else:
            rule_intent = "RESEARCH"

        def clean_input(text: str) -> str:
            return text.lstrip("!?,. ").split("\n")[0].split(" ", 1)[-1] if text.lower().startswith(("hello", "hi ", "hey")) else text

        
        is_feedback = "YES" if self.is_action_feedback(user_input) else "NO"
        
        router_prompt = (
            "You are an intent router.\n"
            "Output ONE word only: CODE or RESEARCH.\n\n"
            f"User input:\n{clean_input(user_input)}\n\n"
            f"Previous intent: {self.last_intent or 'NONE'}\n"
            f"Rule-based suggestion: {rule_intent or 'NONE'}\n"
            f"Signal: feedback_or_acknowledgment={is_feedback}\n\n"
            "Policy:\n"
            "- The rule-based suggestion is the default decision.\n"
            "- Override the rule-based suggestion ONLY if it is CLEARLY wrong.\n"
            "- If you are uncertain, keep the rule-based suggestion.\n\n"
            "Definitions:\n"
            "- CODE = user wants code OR technical implementation details (GUI/app, Electron, API, debugging, scripts, setup).\n"
            "- RESEARCH = explanations, discussion, advice, planning without code.\n\n"
            "Hard rules:\n"
            "- If message is ONLY thanks/ok/feedback with no new request -> RESEARCH.\n"
            "- If mixed feedback + new request -> classify the request part.\n\n"
            "- Ignore all the user politenes (Can you, Please, etc.) and greetings(hello, hi, hey)"
            "Repty with ONE word only\n\n"
        )

        
        
        decision = self._call_ollama(self.router_model, router_prompt, target_temp = self.temps["ROUTER"], target_context = self.context_size["ROUTER"])
        intent = decision.strip().upper()

        if rule_intent and rule_intent != intent:
            print(f"[Router Override] Rule={rule_intent} → LLM={intent}")

        if intent not in {"CODE", "RESEARCH"}:
            intent = rule_intent or "RESEARCH"


        self.last_intent = intent

        return intent

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