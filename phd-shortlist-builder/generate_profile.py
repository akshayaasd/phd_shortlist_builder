"""
Profile Generator — compiles raw resume text and details into a validated student profile JSON.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from openai import AsyncOpenAI
import click
import asyncio

from src.llm_client import LLMClient
from src.profile_parser import load_profile

async def parse_resume_to_json(resume_text: str, client: LLMClient) -> dict:
    """Uses LLM to structure raw resume text into JSON."""
    prompt = (
        "Parse the following raw resume/CV text into a structured JSON object containing "
        "education, skills, projects, and publications. "
        "You must return ONLY a JSON object with this exact structure:\n"
        "{\n"
        '  "education": [\n'
        "    {\n"
        '      "degree": "B.S. / M.S. / Ph.D. / etc",\n'
        '      "field": "Field of study",\n'
        '      "institution": "University name",\n'
        '      "graduation_year": 2024,\n'
        '      "gpa": "GPA value or null",\n'
        '      "thesis": "Thesis title or null"\n'
        "    }\n"
        "  ],\n"
        '  "skills": ["Skill1", "Skill2"],\n'
        '  "projects": [\n'
        "    {\n"
        '      "title": "Project Title",\n'
        '      "description": "Short description"\n'
        "    }\n"
        "  ],\n"
        '  "publications": [\n'
        "    {\n"
        '      "title": "Paper Title",\n'
        '      "venue": "Venue/Journal",\n'
        '      "year": 2023\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"Resume text:\n{resume_text}"
    )

    try:
        click.echo("Structuring resume text using LLM...")
        return await client.chat_complete_json(prompt)
    except Exception as e:
        click.echo(f"LLM parsing failed: {e}. Falling back to default empty template.")
        return {
            "education": [],
            "skills": [],
            "projects": [],
            "publications": []
        }

@click.command()
@click.option("--resume-path", help="Path to a text file containing the raw resume/CV.")
@click.option("--output-path", default="sample_profiles/new_student.json", help="Path where JSON will be written.")
def main(resume_path: str | None, output_path: str):
    """Interactively build a validated student profile JSON."""
    click.echo("=========================================")
    click.echo("PhD Shortlist Builder — Profile JSON Generator")
    click.echo("=========================================")

    # Initialize client
    openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", "sk-placeholder"))
    llm_client = LLMClient(openai_client)

    # 1. Get raw resume
    raw_resume = ""
    if resume_path:
        raw_resume = Path(resume_path).read_text(encoding="utf-8")
    else:
        click.echo("\nPlease paste the raw resume/CV text (or press Enter to import from interactive fields):")
        lines = []
        while True:
            try:
                line = input()
                if not line and lines and lines[-1] == "":
                    # Double enter finishes
                    break
                lines.append(line)
            except EOFError:
                break
        raw_resume = "\n".join(lines).strip()

    # 2. Parse structured fields
    parsed_data = {}
    if raw_resume and (llm_client.use_openai or llm_client.use_gemini):
        parsed_data = asyncio.run(parse_resume_to_json(raw_resume, llm_client))
    else:
        # Template defaults
        parsed_data = {
            "education": [
                {
                    "degree": "B.Tech",
                    "field": "Biomedical Engineering",
                    "institution": "IIT Delhi",
                    "graduation_year": 2022,
                    "gpa": "8.9/10",
                    "thesis": "Non-invasive neural signal decoding using EEG"
                }
            ],
            "skills": ["Python", "PyTorch"],
            "projects": [],
            "publications": []
        }

    # 3. Interactive inputs for other fields
    click.echo("\n--- Stated Details ---")
    student_id = input("Student ID (default: student_new): ").strip() or "student_new"
    
    interests_raw = input("Research Interests (comma separated, e.g. computational neuroscience, brain-computer interfaces): ").strip()
    research_interests = [i.strip() for i in interests_raw.split(",") if i.strip()] if interests_raw else ["computational neuroscience"]
    
    countries_raw = input("Target Countries (comma separated ISO codes, e.g. US, UK, CA): ").strip() or "US, UK, CA"
    target_countries = [c.strip().upper() for c in countries_raw.split(",") if c.strip()]
    
    semester = input("Target Intake Semester (default: Fall): ").strip() or "Fall"
    year_raw = input("Target Intake Year (default: 2026): ").strip() or "2026"
    target_intake = {"semester": semester, "year": int(year_raw) if year_raw.isdigit() else 2026}
    
    nationality = input("Nationality Country Code (default: IN): ").strip().upper() or "IN"
    intro_call_summary = input("Intro Call Summary (free text background info): ").strip() or "Interested in theoretical and experimental neuroscience."

    # Assemble profile
    profile_json = {
        "student_id": student_id,
        "education": parsed_data.get("education", []),
        "skills": parsed_data.get("skills", []),
        "projects": parsed_data.get("projects", []),
        "publications": parsed_data.get("publications", []),
        "research_interests": research_interests,
        "target_countries": target_countries,
        "target_intake": target_intake,
        "nationality": nationality,
        "intro_call_summary": intro_call_summary,
        "raw_resume": raw_resume
    }

    # Save and validate
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(profile_json, indent=2), encoding="utf-8")

    try:
        load_profile(out_file)
        click.echo(f"\nSuccess! Profile generated and validated at: {out_file}")
    except Exception as e:
        click.echo(f"\nWarning: Profile saved but validation failed: {e}")

if __name__ == "__main__":
    main()
