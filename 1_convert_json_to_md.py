import json
from pathlib import Path
import os

# --- Configuration ---
# The folder where your profile is located and where new files will be created.
CONSTITUTION_DIR = Path(__file__).parent / "personal_constitution"
# The name of your original JSON file.
JSON_FILE_PATH = CONSTITUTION_DIR / "魏玉彬_baseline_profile.json"

def convert_json_to_markdown():
    """
    Reads the baseline JSON profile and converts it into multiple,
    structured Markdown files.
    """
    print(">> Starting conversion process...")
    
    # Create output directories if they don't exist
    career_dir = CONSTITUTION_DIR / "01_Career_Experience"
    project_dir = CONSTITUTION_DIR / "02_Project_Experience"
    os.makedirs(career_dir, exist_ok=True)
    os.makedirs(project_dir, exist_ok=True)
    
    print(f"   Reading data from: {JSON_FILE_PATH.name}")
    try:
        with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ ERROR: The file was not found at {JSON_FILE_PATH}")
        print("   Please make sure the file exists and the name is correct.")
        return
    except json.JSONDecodeError:
        print(f"❌ ERROR: The file {JSON_FILE_PATH.name} is not a valid JSON file.")
        return

    # --- 1. Create the main Constitution File ---
    print("   -> Creating 00_MY_CONSTITUTION.md...")
    with open(CONSTITUTION_DIR / "00_MY_CONSTITUTION.md", "w", encoding="utf-8") as f:
        f.write("# My Personal Constitution\n\n")
        
        # Write Self Assessment
        assessment = data.get("self_assessment_summary", "No self-assessment provided.")
        f.write("## Self Assessment Summary\n")
        f.write(assessment.replace("【自我评价】\n", "").replace("【附加消息】\n", "\n## Additional Information\n"))
        f.write("\n\n")

        # Write Skills
        skills = data.get("skills_summary", {})
        f.write("## Skills Summary\n")
        f.write(f"- **Technical Skills:** {skills.get('technical_skills', 'N/A')}\n")
        for lang in skills.get('languages', []):
            f.write(f"- **Language:** {lang.get('language')} ({lang.get('proficiency')})\n")
        f.write("\n")

    # --- 2. Create Work Experience Files ---
    print(f"   -> Processing {len(data.get('work_experience', []))} work experiences...")
    for i, job in enumerate(data.get("work_experience", [])):
        company = job.get("company_name", "Unknown Company").replace(" ", "_")
        start_date = job.get("start_date", "nodate")
        filename = f"{start_date}_{company}.md"
        
        with open(career_dir / filename, "w", encoding="utf-8") as f:
            f.write(f"# Work Experience: {job.get('position_title')} at {job.get('company_name')}\n")
            f.write(f"**Period:** {job.get('start_date')} to {job.get('end_date')}\n\n")
            
            f.write("## Job Description\n")
            f.write(f"{job.get('job_description', 'N/A')}\n\n")
            
            f.write("## Achievements & Reflections\n")
            f.write(f"{job.get('achievements', 'No specific achievements listed.')}\n")
    
    # --- 3. Create Project Experience Files ---
    print(f"   -> Processing {len(data.get('project_experience', []))} project experiences...")
    for i, proj in enumerate(data.get('project_experience', [])):
        project_name = proj.get("project_name", "Unknown Project").replace(" ", "_").replace("/", "_")
        start_date = proj.get("start_date", "nodate")
        filename = f"{start_date}_{project_name}.md"

        with open(project_dir / filename, "w", encoding="utf-8") as f:
            f.write(f"# Project Experience: {proj.get('project_name')}\n")
            f.write(f"**Role:** {proj.get('role_in_project', 'N/A')}\n")
            f.write(f"**Period:** {proj.get('start_date')} to {proj.get('end_date')}\n\n")
            
            f.write("## Project Description\n")
            f.write(f"{proj.get('project_description', 'N/A')}\n\n")

    print("\n✅ Conversion complete! Your new Markdown files have been created.")
    print("   You can now delete the old JSON file if you wish.")

if __name__ == "__main__":
    convert_json_to_markdown()
