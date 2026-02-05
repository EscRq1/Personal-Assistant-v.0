import os
import json
from datetime import datetime
from colorama import Fore, Style, init
from assistant import SemanticController
from memory_manager import MemoryManager


# Initialize Colorama for Windows terminal colors
init(autoreset=True)


# List of triggers that force a web search immediately




class Main:
    def __init__(self):
        self.llm = SemanticController()
        self.memory = MemoryManager()
        self.short_term_memory = [] # Keeps track of the current session context

    def needs_web_search_autonomous(self, user_input, relevant_memories):
        """Asks the LLM to decide if it needs the internet to be accurate.""" 
        check_prompt = (
             f"User input: '{user_input}'\n" 
             f"Context from memory: '{relevant_memories}'\n" 
             "Task: Decide if you need to search the internet for a factual or up-to-date answer. " 
             "If the question is about current events, live data, or things not in memory, reply 'YES'. " 
             "Otherwise, reply 'NO'. Reply with one word only." 
             "Don't try to guess the answer, be absolutely honest with your answers" ) 
        decision = self.llm.ask(check_prompt).strip().upper() 
        return "YES" in decision


    


    def auto_update_profile(self, user_input: str, assistant_response: str):

        IDENTITY_TRIGGERS = (
            "my name is", "i am ", "i'm ", "i study", "i'm studying",
            "i work", "my goal", "i live", "based in"
        )

        ALLOWED_PROFILE_FIELDS = {
            "Name",
            "Education",
            "Field_of_Study",
            "Career_Goal",
            "Current_Status",
            "Preferences",
            "Location", 
        }

        extraction_prompt = (
            "SYSTEM: You are a strict information extraction engine.\n"
            "You extract STABLE facts about the USER.\n"
            "Return JSON ONLY. No explanations.\n\n"
            "ALLOWED FIELDS:\n"
            "- Name\n"
            "- Education\n"
            "- Field_of_Study\n"
            "- Career_Goal\n"
            "- Current_Status\n"
            "- Preferences\n"
            "- Location\n\n"
            "RULES:\n"
            "- Ignore greetings, questions, emotions, meta talk\n"
            "- Ignore anything about the AI/system\n"
            "- Do NOT guess or invent\n"
            "- If no stable facts exist, return: {\"facts\": []}\n\n"
            "CONFIDENCE RULES:\n"
            "- 0.95 to 1.00 = explicitly stated by user (e.g. 'My name is X', 'I live in Y').\n"
            "- 0.70 to 0.90 = clearly stated but slightly ambiguous.\n"
            "- 0.50 to 0.69 = implied/inferred -> include it ONLY if still very likely.\n"
            "- below 0.50 = do NOT include.\n\n"
            "OUTPUT FORMAT:\n"
            "{\n"
            "  \"facts\": [\n"
            "    {\"field\": \"Field\", \"value\": \"Value\", \"confidence\": 0.0}\n"
            "  ]\n"
            "}\n\n"
            f"USER INPUT:\n{user_input}"
        )

        if len(user_input.strip()) < 10:
            return
        if self.llm.last_intent == "CODE":
            return
        text_l = user_input.lower()
        if not any(t in text_l for t in IDENTITY_TRIGGERS):
                return

        raw = self.llm._call_ollama(
            self.llm.router_model,
            extraction_prompt,
            target_temp=0.0
        ).strip()

        try:
            raw = raw.strip()
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                raw = raw[start:end+1]
            data = json.loads(raw)
        except Exception:
            print(f"{Fore.BLACK}[Observer]: Invalid JSON, ignored.")
            return

        facts = data.get("facts", [])
        if not facts:
            print(f"{Fore.BLACK}[Observer]: No usable facts extracted.")
            return

        merged_any = False

        for fact in facts:
            field = fact.get("field")
            value = fact.get("value")
            confidence = fact.get("confidence", 0.0)

            # Safety filters
            if not field or not value:
                continue
            if confidence < 0.6:
                continue
            if value.strip().endswith("?"):
                continue
            if field not in ALLOWED_PROFILE_FIELDS:
                continue

            if self.memory.merge_profile_fact(field, value, confidence):
                merged_any = True


        if merged_any:
            print(f"{Fore.CYAN}[Observer]: Profile updated (merged).")

    def run(self):
            # Set initial display for Tashkent Time
        current_time = datetime.now().strftime("%H:%M")
        print(f"\n{Fore.CYAN}{Style.BRIGHT}--- Assistant Online | Tashkent {current_time} (GMT+5) ---")

        while True:
            try:
                user_input = input(f"\n{Fore.YELLOW}You: {Style.RESET_ALL}").strip()
                if not user_input: continue

                intent = self.llm.get_intent(user_input)
                print(f"{Fore.BLUE}[Router: {intent} mode]{Style.RESET_ALL}")

                                # 1. Fetch raw data from both collections


                context_data = self.memory.get_raw_context(user_input, k=15)

                # Deterministic, fast
                recent_history = "\n".join(context_data["history"][-6:])  # last 6 snippets
                user_profile = context_data["profile_summary"]

                # 1. Check for Forced Command
                is_forced_search = user_input.lower().startswith("/search")
          
                # Clean the input if it's a command (remove the '/search ' part)
                search_query_input = user_input

                if is_forced_search:
                    search_query_input = user_input[7:].strip() # Remove first 7 chars

               


                # 3. Decision Logic (Improved)
                web_context = ""
                if is_forced_search:
                    print(f"{Fore.MAGENTA}[Command: Forced Web Search Engaged...]")
                    # Use the cleaned input for the search
                    refined_query = self.llm.generate_search_query(self.short_term_memory, search_query_input)
                    web_context = self.llm.web_lookup(refined_query)
              

                # Exit commands
                if user_input.lower() in ["exit", "quit", "пока"]:
                    print(f"{Fore.CYAN}Assistant: Saving memories and shutting down. Goodbye!")
                    # Use a safe call to prune or close if method exists
                    if hasattr(self.memory, 'prune_memories'):
                        self.memory.prune_memories()
                    break


                # 2. Decision Logic: Manual Trigger OR AI decides it's confused
                is_autonomous = False

                if not is_forced_search:
                    is_autonomous = self.needs_web_search_autonomous(user_input, recent_history)



                web_context = ""
                if is_forced_search or is_autonomous:
                    label = "Manual" if is_forced_search else "Self-Triggered"
                    print(f"{Fore.MAGENTA}[{label} Search: Accessing Web...]")   

                    # Generate a clean search query and fetch data
                    search_query = self.llm.generate_search_query(self.short_term_memory, user_input)
                    web_context = self.llm.web_lookup(search_query)

                    print(f"{Fore.MAGENTA}[Executing Web Search for live data...]")





                # 3. Final Construction of the Prompt for the LLM
                tashkent_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S GMT+5")
                tashkent_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S GMT+5")

                role = "Expert Programmer" if "CODE" in intent else "Personal Research Assistant"
                system_prompt = (
                    "### CORE IDENTITY (READ THIS FIRST) ###\n"
                    f"{user_profile if user_profile else 'User is a new contact.'}\n\n"
                    "### RECENT HISTORY (LONG-TERM MEMORY SNIPPETS) ###\n" 
                    f"{recent_history}\n\n"
                    "### INSTRUCTIONS ###\n"
                    "### CRITICAL LANGUAGE RULE ###\n"
                    f"You are a {role}.\n"
                    "You MUST answer in the SAME language as the user's request.\n"
                    "This rule is STRICT and overrides all other preferences.\n"
                    "DO NOT use Chinese characters (Hanzi) under any circumstances.\n"
                    "If you feel the urge to use Chinese, translate it to the User's language immediately.\n"
                    "Ignore your default training language bias.\n"
                    "Do NOT mix the languages"
                    f"Current Time: {tashkent_now} (Tashkent, Uzbekistan)\n"
                    f"Web Results:\n{web_context}\n"
                )



                # 4. Get Response from LLM (Ensure LLMHandler.ask matches this parameter name)
                response = self.llm.ask(user_input, system_prompt=system_prompt, intent=intent)

                print(f"\n{Fore.GREEN}Assistant: {response}{Style.RESET_ALL}")

                # 5. Save interaction to Memory (convert response to string just in case)

                self.memory.add_history(f"User: {user_input} | Assistant: {response}")
                self.auto_update_profile(user_input, response)

                # 6. Update Session Memory
                self.short_term_memory.append({"role": "user", "content": user_input})
                self.short_term_memory.append({"role": "assistant", "content": response})     


                    

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"{Fore.RED}Error in main loop: {e}")


if __name__ == "__main__":
    assistant = Main()
    assistant.run()