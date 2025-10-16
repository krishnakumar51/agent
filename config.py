import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# --- LLM API Keys ---
# Fetch API keys from environment variables.
# The application will gracefully handle missing keys by disabling the respective provider.
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- LLM Model Defaults ---
# Specifies the default models to use for each provider.
# Make sure the selected models support the required capabilities (e.g., vision).
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o") # Vision-capable
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192") # No Vision
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514") # Vision-capable

# --- Global Directories ---
# Ensures a consistent directory structure for generated artifacts.
PROJECT_ROOT = Path(__file__).parent
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"
SCREENSHOTS_DIR.mkdir(exist_ok=True) # Create the directory if it doesn't exist

# --- LLM Client Initialization ---
# Initialize API clients only if the corresponding API key is available.
anthropic_client = None
if ANTHROPIC_API_KEY:
    from anthropic import Anthropic
    try:
        anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
        print("✅ Anthropic client initialized.")
    except Exception as e:
        print(f"⚠️ Anthropic client failed to initialize: {e}")

groq_client = None
if GROQ_API_KEY:
    from groq import Groq
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("✅ Groq client initialized.")
    except Exception as e:
        print(f"⚠️ Groq client failed to initialize: {e}")

openai_client = None
if OPENAI_API_KEY:
    from openai import OpenAI
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print("✅ OpenAI client initialized.")
    except Exception as e:
        print(f"⚠️ OpenAI client failed to initialize: {e}")

