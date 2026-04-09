import google.generativeai as genai
import os
from dotenv import load_dotenv
from pathlib import Path
import shutil

# --- Initialization & Setup ---
load_dotenv()
API_KEY = os.getenv("GOOGLE_AI_KEY")

# --- Configuration ---
# Define the paths for our workflow
BASE_DIR = Path(__file__).parent
CONSTITUTION_DIR = BASE_DIR / "personal_constitution"
DRAFTS_DIR = CONSTITUTION_DIR / "_drafts"
FORMATTED_DIR = CONSTITUTION_DIR / "_formatted_for_review"
ARCHIVE_DIR = DRAFTS_DIR / "archive"

# --- API Initialization ---
if not API_KEY or "..." in API_KEY:
    print("❌ Google AI API Key is not configured in your .env file.")
    exit()

try:
    genai.configure(api_key=API_KEY)
    gemini_model = genai.GenerativeModel("models/gemini-pro-latest")
    print("✅ Gemini API initialized successfully.")
except Exception as e:
    print(f"❌ Failed to initialize Gemini API: {e}")
    exit()


def process_single_draft(file_path):
    """
    Takes a single draft file, sends its content to the AI for formatting,
    saves the result, and archives the original.
    """
    print(f"\nProcessing draft: {file_path.name}")

    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    if not raw_text.strip():
        print("   - File is empty. Archiving.")
        shutil.move(file_path, ARCHIVE_DIR / file_path.name)
        return

    prompt = f"""
# ROLE
You are a Personal Archivist AI. Your task is to take the user's raw, unstructured text and organize it into a clear, structured Markdown document using the S.A.R.L. framework (Situation, Action, Result, Lesson Learned). If the text is a simple diary entry or reflection, just give it a good title and format it cleanly.

# INSTRUCTIONS
1.  Read the user's raw text below.
2.  Analyze its content. If it seems like a professional experience, structure it with the Markdown headings: `## Situation`, `## My Actions`, `## Result`, `## Lessons Learned & Personal Reflections`.
3.  If it seems like a personal diary or a random thought, simply format it with a relevant `# Title` and clean paragraphs.
4.  Rewrite the content in clear, concise English.
5.  Your output must be only the formatted Markdown content. Do not add any extra commentary.

# USER'S RAW TEXT
---
{raw_text}
---
"""

    try:
        print("   -> Sending to Gemini for formatting...")
        response = gemini_model.generate_content(prompt)
        formatted_text = response.text.strip()

        # Save the formatted text to the review folder
        output_path = FORMATTED_DIR / file_path.name
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(formatted_text)
        print(f"   -> Saved formatted version to: {output_path.relative_to(BASE_DIR)}")

        # Move the original draft to the archive
        shutil.move(file_path, ARCHIVE_DIR / file_path.name)
        print(f"   -> Archived original draft.")

    except Exception as e:
        print(f"❌ An error occurred while processing {file_path.name}: {e}")


if __name__ == "__main__":
    print("🚀 Starting the AI Draft Processing Workflow...")

    # Find all text or markdown files in the drafts directory
    draft_files = [
        f for f in DRAFTS_DIR.iterdir() if f.is_file() and f.suffix in [".txt", ".md"]
    ]

    if not draft_files:
        print("No new drafts to process. The '_drafts' folder is empty.")
    else:
        print(f"Found {len(draft_files)} draft(s) to process.")
        for draft_file in draft_files:
            process_single_draft(draft_file)
        print("\n✅ All drafts have been processed.")
