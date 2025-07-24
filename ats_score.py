import google.generativeai as genai
import re
import json

genai.configure(api_key="AIzaSyDXlJv-M3hiw-kLSWfpd0xQgk4Ssz7G8Fo")
model = genai.GenerativeModel('gemini-1.5-flash')

def calculate_score(parsed_output, format_score, grammatical_score):
    prompt = f"""
You are an AI-powered resume evaluation engine for an advanced ATS system.

You must evaluate the following **six** categories, each scored out of 100:
- Skills
- Experience
- Projects_Quality
- Certifications
- Format_Score (value strictly provided by the system — do NOT evaluate, only copy from input)
- Grammatical_Score (value strictly provided by the system — do NOT evaluate, only copy from input)

### Scoring Guidelines:

1. Skills: Assess breadth, depth, and alignment with target roles. Score higher for a wider set of relevant, high-demand skills.
2. Experience: Base the score on relevance, impact, complexity, progression, and clear accomplishments in professional roles.
3. Projects_Quality:
   - Analyze all projects. For each, assess the technical challenge, real-world impact, originality, technology stack, scale, and outcome.
   - Assign higher scores for advanced, complex, innovative, or large-scale projects; lower scores for basic, academic, or repetitive projects.
   - Raise scores if projects use cutting-edge technologies, solve real-world problems, or produce measurable outcomes.
   - Lower scores if projects are generic, lack detail, have unclear outcomes, or only repeat simple coursework.
4. Certifications:
   - Score higher for certifications from globally recognized organizations (e.g., AWS, Google, Microsoft, Oracle, PMP, ISACA, etc.), industry-standard exams, or advanced technology specializations.
   - Give intermediate scores to widely accepted but not top-tier certifications (such as Coursera/edX advanced tracks, recognized university-short courses).
   - Score lower for entry-level or generic certifications, online course participation, or overly common/unaccredited certificates.
5. Format_Score: Value strictly provided by the system—do NOT evaluate, only use the input as given.
6. Grammatical_Score: Value strictly provided by the system—do NOT evaluate, only use the input as given.

For all categories, judge each item by its **quality, prestige, level of technical or professional challenge, and real-world significance**. Internships should be scored based on the selectivity of the company/organization, level of responsibility, technology used, and impact delivered—distinguishing between top-tier, mid-tier, and generic internship experiences.

**Do NOT hallucinate or invent items. Evaluate only what is explicitly present. Your evaluation must always be strictly aligned with this rubric.**


### JSON OUTPUT ONLY (no markdown, no explanations):

{{
  "scores": {{
    "Skills":    {{"score": X, "comment": "..."}},
    "Experience":    {{"score": X, "comment": "..."}},
    "Projects_Quality": {{"score": X, "comment": "..."}},
    "Certifications":   {{"score": X, "comment": "..."}},
    "Format_Score": {format_score},
    "Grammatical_Score": {grammatical_score},
    "weighted_total": X
  }},
  "evaluation": "Brief summary of key strengths and areas for improvement."
}}

### You must use this weighted formula (round final result to integer):

weighted_total = (Skills * 0.3) + (Experience * 0.2) + (Projects_Quality * 0.2) + (Certifications * 0.15) + (Format_Score * 0.1) + (Grammatical_Score * 0.05)

Weighted_total must always use the above weights, regardless of input or content.

###
Now evaluate the following resume content (input is below):

{parsed_output}

Format_Score = {format_score}
Grammatical_Score = {grammatical_score}
"""

    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(temperature=0.0, top_p=1.0, top_k=1)
    )
    raw_text = response.text.strip()
    try:
        data = json.loads(raw_text)
    except Exception:
        # Regex fallback
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if json_match:
            json_string = json_match.group(0)
            data = json.loads(json_string)
        else:
            print(f"❌ JSON not detected.\n\nRaw response:\n{raw_text}")
            return None

    # Post-process / verify result
    # If weights or values are off, recompute locally
    try:
        skills = max(0, min(100, data["scores"]["Skills"]["score"]))
        experience = max(0, min(100, data["scores"]["Experience"]["score"]))
        projects = max(0, min(100, data["scores"]["Projects_Quality"]["score"]))
        certifications = max(0, min(100, data["scores"]["Certifications"]["score"]))
        format_sc = max(0, min(100, format_score))
        grammar_sc = max(0, min(100, grammatical_score))

        weighted_total = int(round(
            skills*0.3 +
            experience*0.2 +
            projects*0.2 +
            certifications*0.15 +
            format_sc*0.10 +
            grammar_sc*0.05
        ))
        # Force overwrite if LLM got it wrong
        data["scores"]["weighted_total"] = weighted_total
    except Exception as e:
        print(f"❌ Post-processing error: {e}\nRAW: {data}")
        return None

    print(json.dumps(data, indent=2))
    return data["scores"]["weighted_total"]
