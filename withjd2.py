import google.generativeai as genai
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet
import nltk

# Ensure WordNet is available
try:
    wordnet.synsets('test')
except:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# Configure Gemini API (replace with your API key)
genai.configure(api_key="AIzaSyDXlJv-M3hiw-kLSWfpd0xQgk4Ssz7G8Fo")

# ---------------- Synonym Expansion ----------------
def get_synonyms(skill):
    skill = skill.lower()
    synonyms = set([skill])
    for syn in wordnet.synsets(skill):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' ').lower())
    return synonyms

# ---------------- JD Skill Extraction (Gemini) ----------------
def extract_skills_gemini(jd_text):
    prompt = f"""
Analyze the following Job Description (JD) and extract ALL relevant skills with the highest possible accuracy.

Follow these strict rules to ensure precise extraction and weighting:

1. **Comprehensive Skill Extraction**:
   - Include technical, non-technical, and soft skills (explicitly stated or implied).
   - Infer hidden skills (e.g., "optimize performance" → "Performance Optimization"; "integrate APIs" → "REST API Integration").
   - Normalize all acronyms and plurals (e.g., "DRF" → "Django REST Framework", "APIs" → "API").

2. **Deduplication with Semantic Grouping (Cosine Similarity)**:
   - Group semantically similar skills (e.g., "React" and "ReactJS", "teamwork" and "team collaboration") into one canonical skill.
   - Assume we will use **vector embeddings with cosine similarity** to merge skills: treat any skills with >0.8 similarity as identical.
   - Always output the **canonical name** (most common/standard form).

3. **Categorization**:
   - Each skill must be labeled as one: Technical, Non-Technical, or Soft.

4. **Priority and Weighting**:
   - **Mandatory**: explicitly marked "must-have", "required", "essential", or repeated multiple times.
   - **Preferred**: "preferred", "plus", "nice-to-have".
   - **Optional**: mentioned casually.
   - Assign a numeric weight from 1–5:
     - 5 = Core, mandatory, highly critical.
     - 4 = Strongly preferred or frequently used.
     - 3 = Useful but secondary.
     - 2 = Nice-to-have.
     - 1 = Optional or bonus.

5. **Context Sensitivity**:
   - Boost weights for skills mentioned in **"Key Responsibilities"**, **"Required Skills"**, or **"Tech Stack"** sections.
   - Reduce weights for general soft skills unless they are emphasized as must-haves.

Return ONLY valid JSON in this format (no extra text):

{{
  "skills": [
    {{
      "name": "Skill Name",
      "category": "Technical/Non-Technical/Soft",
      "priority": "Mandatory/Preferred/Optional",
      "weight": 1-5
    }}
  ]
}}

Now analyze this Job Description and output the JSON:
{jd_text}
"""

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    output = response.text.strip()
    json_match = re.search(r'\{[\s\S]*\}', output)
    return json_match.group(0) if json_match else "{}"

# ---------------- AI Score Calculation ----------------
def calculate_ai_score(candidate, jd_text):
    candidate_text = json.dumps(candidate, indent=2)
    prompt = f"""
You are an expert technical recruiter. Your task is to score this candidate (0–100) against the Job Description (JD). 
Your scoring MUST be **deterministic, consistent, and repeatable** — meaning if the same resume and JD are provided multiple times, the score must remain identical (no randomness).

Follow this **EXACT step-by-step method** and NEVER change weights, even slightly, between runs:

--------------------------------------------------------------------
### STEP 1: Identify JD Skills
1. Extract ALL mandatory, preferred, and optional skills from the JD.
2. Normalize skill names (e.g., "React" = "ReactJS").
3. Categorize each skill as Mandatory, Preferred, or Optional.

--------------------------------------------------------------------
### STEP 2: Candidate Skills Matching
1. For each Mandatory JD skill the candidate has → **+5 points**.
2. For each Preferred JD skill the candidate has → **+3 points**.
3. For each Optional JD skill or additional relevant skill not in the JD but valuable → **+1 point**.
4. For each Mandatory JD skill the candidate is missing → **-2 points**.

Keep track of:
- Matched mandatory skills count.
- Matched preferred skills count.
- Extra useful skills.
- Missing mandatory skills.

--------------------------------------------------------------------
### STEP 3: Experience and Projects
1. Award **+10 points** for each work experience entry (job or internship) that involves any JD-required skills.
2. Award **+5 points** for each project that involves JD-required skills.
3. Deduct **-5 points** for experiences or projects that are unrelated or irrelevant to the JD.

If skills are listed in the candidate’s "Skills" section but not explicitly in projects/experience descriptions, 
ASSUME they used those skills during work/projects and award credit proportionally (divide evenly across entries).

--------------------------------------------------------------------
### STEP 4: General Relevance (Soft Skills & Fit)
1. Award 0–10 points based on:
   - Clarity of resume.
   - Presence of leadership, teamwork, problem-solving, adaptability, etc.
   - Overall match to the role based on context.

--------------------------------------------------------------------
### STEP 5: Combine Scores
1. Start with a base score of 0.
2. Add points from Steps 2–4.
3. Cap the total score between 0 and 100.
4. Round to the nearest whole number.

--------------------------------------------------------------------
### RULES FOR CONSISTENCY
- Use EXACT weights as above — do not improvise or adjust based on "gut feeling".
- Do NOT change logic between different runs.
- Follow the rubric strictly even if the candidate seems "overqualified" or "underqualified".
- Always produce the SAME result for the SAME input.

--------------------------------------------------------------------
### OUTPUT FORMAT (STRICT)
- Return ONLY the final numeric score (0–100).
- Do NOT include reasoning, text, explanations, or extra formatting.
- Do NOT output words, only the number.

Now, calculate the score for this candidate.

Job Description:
{jd_text}

Candidate Resume (JSON):
{candidate_text}
"""
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt, generation_config={"temperature": 0})
    score_text = response.text.strip()
    try:
        return float(re.search(r'\d+', score_text).group(0))
    except:
        return 0.0


# ---------------- Final Score Combination (Only AI Score) ----------------
def calculate_final_score(candidate, jd_text):
    ai_score = calculate_ai_score(candidate, jd_text)
    return ai_score  # Only AI score is returned

# ---------------- Main ----------------
if __name__ == "__main__":
    print("Paste the Job Description (JD). Type 'END' when done:")
    jd_text = ""
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        jd_text += line + "\n"

    print("\nPaste the Candidate Resume JSON. Type 'END' when done:")
    candidate_json = ""
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        candidate_json += line + "\n"

    candidate = json.loads(candidate_json)

    final_score = calculate_final_score(candidate, jd_text)
    print(f"\nFinal Candidate Score: {final_score}/100")
