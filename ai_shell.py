import os
import sys
import subprocess
from google import genai
from google.genai import types

try:
    client = genai.Client()
except Exception as e:
    print(f"[!] Warning: Could not initialize Gemini client. Make sure GEMINI_API_KEY is set. Error: {e}")
    client = None

def print_banner():
    print("==================================================")
    print("      AI-POWERED TERMINAL SHELL (GEMINI)          ")
    print("==================================================")
    print(" Type standard shell commands (e.g., ls, pwd)")
    print(" Type 'ai: <your prompt>' to have Gemini write code/commands")
    print(" Type 'exit' to quit")
    print("--------------------------------------------------")

def handle_ai_request(prompt):
    if not client:
        print("[!] Gemini client is not configured. Set your GEMINI_API_KEY.")
        return

    print(f"[*] Asking Gemini to handle: '{prompt}'...")
    
    system_instruction = (
        "You are an expert Linux terminal assistant integrated into a Python custom shell. "
        "When the user asks a request, output ONLY a valid shell command that accomplishes the task. "
        "Avoid complex python -c one-liners with nested quotes if a standard linux utility (like find, wc, grep, etc.) works. "
        "Do not include markdown code block formatting like ```bash or ```, and do not include conversational filler."
    )

    try:
        response = client.models.generate_content(
            model='gemini-3.5-flash-lite',
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1
            )
        )
        command = response.text.strip()
        command = command.replace("```bash", "").replace("```", "").strip()

        print(f"\n[+] Generated Command:\n    {command}\n")
        confirm = input("Execute this command? (y/n/edit): ").strip().lower()
        
        if confirm == 'y':
            subprocess.run(command, shell=True)
        elif confirm == 'edit':
            edited = input("Enter modified command: ").strip()
            if edited:
                subprocess.run(edited, shell=True)
        else:
            print("Command cancelled.")

    except Exception as e:
        print(f"[!] Error communicating with Gemini API: {e}")

def main():
    print_banner()
    while True:
        try:
            cwd = os.getcwd()
            display_cwd = cwd.replace(os.path.expanduser("~"), "~")
            prompt_str = f"ai-shell:{display_cwd}$ "
            user_input = input(prompt_str).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting AI shell...")
            break

        if not user_input:
            continue
        
        if user_input.lower() in ["exit", "quit"]:
            print("Peace out!")
            break

        if user_input.lower().startswith("ai:"):
            ai_prompt = user_input[3:].strip()
            handle_ai_request(ai_prompt)
        else:
            try:
                subprocess.run(user_input, shell=True)
            except Exception as e:
                print(f"Error executing command: {e}")

if __name__ == "__main__":
    main()
