import fitz  # PyMuPDF
import google.generativeai as genai
import os
import json
from ats_score import calculate_score  # Ensure this function accepts parsed JSON text
from external_parameters import analyze_resume
import re
import tempfile


# 🧠 Configure Google Gemini API
genai.configure(api_key="AIzaSyC6Y49gb19VaiKuDOiCiV8RBeQ12TisKOQ")

# 📝 Gemini Prompt Template
PROMPT_TEMPLATE = """
You are a deterministic and highly consistent resume parser and evaluator. Your job is to extract resume content in a structured JSON format and evaluate it with strict, repeatable rules—always producing the same result for the same input.

---

### Step 1: Structured Resume Data Extraction

Extract and organize the candidate's information into the following *exact JSON format*:

{
  "Contact Information": {
    "name": "...",
    "email": "...",
    "phone": "...",
    "linkedin": "...",  // null if missing
    "location": "..."   // null if missing
  },
  "Summary": "...", // 1-3 sentence summary, null if missing
  "Education": [
    {
      "institution": "...",
      "degree": "...",
      "department": "...",
      "cgpa": "...", // null if missing
      "year_of_completion": "..." // year only, null if missing
    }
  ],
  "Skills": {
    "Languages": [...],
    "Technologies": [...],
    "Core": [...]
  },
  "Certifications": ["..."],  // Empty array if none
  "Projects": [
    {
      "title": "...",
      "date": "...", // null if missing
      "details": ["..."]
    }
  ],
  "Work Experience": [
    {
      "role": "...",
      "organization": "...",
      "location": "...", // null if missing
      "date": "...",
      "responsibilities": ["..."]
    }
  ],
  "Achievements": ["..."],  // Empty array if non
}

---

### Step 2: Resume Evaluation - Atomic-Level Scoring (100 Points Total)
1. SECTION HEADINGS (15 pts)
   15 = All 6 standard sections with exact headers
   12 = 1 non-standard header (e.g., "My Journey")
   9  = 2 non-standard headers
   6  = Missing 1 required section
   3  = Missing 2 required sections
   0  = Missing ≥3 sections

2. ATS PARSE RATE (25 pts)
   25 = Perfect single-column, no tables/graphics
   20 = Minor spacing issues (1-2 instances)
   15 = Non-standard fonts OR 1-2 images
   10 = Single-column tables present
   5  = Multi-column layout detected
   0  = Complex graphics/tables making text unparsable

3. ACTION VERB REPETITION (10 pts)
   10 = No action verbs repeated >1 time
   8  = 1-2 action verbs repeated twice 
   6  = 3-4 action verbs repeated twice
   4  = 5-6 action verbs repeated twice OR 1 verb repeated 3+ times
   2  = 7-8 action verbs repeated twice OR 2 verbs repeated 3+ times
   0  = 9+ repeated action verbs OR 3+ verbs repeated 3+ times

   Action Verbs List (Partial):
   ["managed", "led", "developed", "created", "implemented", 
   "designed", "improved", "increased", "reduced", "optimized",
   "coordinated", "facilitated", "performed", "achieved", "built"]

   Rules:
   - Count ONLY verbs from predefined action verb list
   - Different tenses count as same verb (manage/managed/managing)
   - Must appear in bullet points (ignore summary/headers)
   - Consecutive bullets count as separate instances

4. GRAMMAR/LANGUAGE (15 pts)
   15 = Flawless grammar and consistent tense
   12 = 1-2 minor errors
   9  = 3-4 errors OR 1 tense inconsistency
   6  = 5 errors OR 2 tense inconsistencies
   3  = 6+ errors with poor phrasing
   0  = Unreadable due to language issues

5. BUZZWORD USE (20 pts)
   20 = All buzzwords supported by quantifiable results
   15 = 1 unsupported buzzword
   10 = 2 unsupported buzzwords
   5  = 3 unsupported buzzwords
   0  = 4+ unsupported buzzwords

6. ACTIVE VOICE (15 pts)
   15 = 90%+ active voice
   12 = 80-89% active voice
   9  = 70-79% active voice
   6  = 60-69% active voice
   3  = 50-59% active voice
   0  = <50% active voice

### SCORING RULES:
- Always round DOWN to nearest integer
- Identical errors → identical deductions
- Count ALL instances (no subjective exceptions)
- Re-verify ambiguous cases against original text

---

### 🔒 Score Consistency & Anti-Hallucination Rules

- The "Total Score" must be calculated as the exact sum of the section scores below, with no additional, missing, or hallucinated values.
- You are strictly forbidden from inventing or adjusting any of the section subscores.
- Your scoring must be deterministic: for any given input, the output score and breakdown will always be identical.
- No creative, intuitive, or inferred adjustments are permitted. Only literal, algorithmic computation per this specification.
- Example: If section scores are 12, 25, 8, 15, 20, 15, the only valid "Total Score" is 95. If this sum exceeds 100, output exactly 100. If it is less than 0, output 0. Never hallucinate a total not represented in the breakdown.
- If context is ambiguous, choose the outcome most directly following these precise instructions.
- Your output JSON must include only:
  - The section subscores and feedback
  - The sum as "Total Score"
- If you make a calculation, always re-add to check the exact total before output.
- If for any reason you cannot calculate a score as specified, output "Total Score": null.

---

### Step 3: Output Format

Return only this JSON structure — no extra comments, headings, or notes:

{
  "Extracted Data": { ... },
  "Miscellaneous Score": {
    "Section Headings": { "score": ..., "feedback": "..." },
    "ATS Parse Rate": { "score": ..., "feedback": "..." },
    "Repetition": { "score": ..., "feedback": "..." },
    "Grammar and Language": { "score": ..., "feedback": "..." },
    "Buzzwords": { "score": ..., "feedback": "..." },
    "Active Voice": { "score": ..., "feedback": "..." }
  },
  "Total Score": (sum of the six section numeric "score" fields above; must match exactly; strictly capped at 100; never hallucinate or infer)
}

---

### Final Consistency Rules (Non-Negotiable):

- No subjective judgment: Only match phrases, words, and formats against predefined lists or patterns.
- Passive voice is detected only if it matches strict regex: (was|were|been|being) \w+ed
- Buzzwords only count if directly followed by a number or measurable outcome (e.g., "reduced by 10%", "delivered 3 projects").
- Repetition is case-insensitive exact match. "Developed" ≠ "develops".
- Grammar deductions only come from a fixed error list. Do not deduct for tone, wordiness, or preferences.
- Do not infer—extract what is literally present in the text.
- Use a static list of penalized buzzwords: ["team player", "go-getter", "self-starter", "passionate"]
- All score deductions are binary—either applied or not, no partial scores.
- Reuse the same tokenizer and regex engine for every parse to prevent platform-level tokenization drift.
- Output must be *bit-for-bit identical* for identical input strings.

---

Now, analyze the resume below:

{resume_text}
"""

def extract_text_from_pdf(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    resume_text = "\n".join(page.get_text() for page in doc)
    return resume_text

def parse_resume_with_gemini(resume_text):
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = PROMPT_TEMPLATE.replace("{resume_text}", resume_text)
    response = model.generate_content(prompt)
    clean_text = response.text.strip()
    # Remove code block markers if present
    if clean_text.startswith("```"):
        clean_text = re.sub(r"^```[a-zA-Z]*\n?", "", clean_text)
        clean_text = re.sub(r"```$", "", clean_text)
    return clean_text.strip()

def process_resume(file_path, job_description=None):
    # Read PDF file as bytes
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    # Extract text from PDF
    print("🔍 Extracting text from PDF...")
    resume_text = extract_text_from_pdf(file_bytes)

    # Parse resume with Gemini
    print("🤖 Parsing resume with Gemini...")
    parsed_json_text = parse_resume_with_gemini(resume_text)
    try:
        parsed_output = json.loads(parsed_json_text)
    except Exception as e:
        print("⚠️ Failed to parse JSON output.")
        print(f"Raw Output:\n{parsed_json_text}")
        print(f"Error: {e}")
        return

    print("✅ Resume successfully parsed!")
    print("🧾 Parsed Resume (JSON):")
    print(json.dumps(parsed_output, indent=2, ensure_ascii=False))


    # Output Score Breakdown (Total Score)
    total_score = parsed_output.get("Total Score", None)
    if total_score is not None:
        print(f"\n**Score Breakdown:** {total_score}/100")

    # Extract scores for ATS calculation
    misc = parsed_output.get("Miscellaneous Score", {})
    format_score = misc.get("ATS Parse Rate", {}).get("score", 0)
    grammatical_score = misc.get("Grammar and Language", {}).get("score", 0)

    # ATS Score
    print("\n📊 ATS Score:")
    calculate_score(
        parsed_output.get("Extracted Data", {}),
        format_score,
        grammatical_score,
        skills_weight=0.3,
        achievement_weight=0.15,
        experience_weight=0.2,
        project_weight=0.2,
        certification_weight=0.1
    )

    # Job Description Match
    if job_description:
        print("\n🔍 Job Description Match Score:")
        jd_score = get_jd_match_score(job_description, parsed_output, "AIzaSyDXlJv-M3hiw-kLSWfpd0xQgk4Ssz7G8Fo")
        print(f"**Match Score:** {jd_score}/100")

# === Example Usage ===
# process_resume("/path/to/your_resume.pdf", job_description="The JD goes here")

if __name__ == "__main__":
    pdf_path = input("Enter the path to your resume PDF: ").strip()
    analyze_resume(pdf_path, bullet_threshold=20)
    jd_text = input("Enter the job description (or leave blank to skip JD matching): ").strip()
    process_resume(pdf_path, job_description=jd_text if jd_text else None)
