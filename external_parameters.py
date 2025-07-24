import os
import re
import fitz  # PyMuPDF
import tempfile

def analyze_resume(file_path):
    results = {}

    # 1. File Info
    file_type = os.path.splitext(file_path)[1]
    file_size = os.path.getsize(file_path) / (1024 * 1024)
    results["file_type"] = file_type
    results["file_size_mb"] = round(file_size, 2)

    # 2. Read PDF
    doc = fitz.open(file_path)
    full_text = ""
    fonts = set()
    images = 0
    lines = 0

    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if b['type'] == 0:
                for l in b['lines']:
                    for s in l['spans']:
                        fonts.add((s['font'], s['size']))
                        full_text += s['text'] + " "
            elif b['type'] == 1:
                images += 1
        lines += len(page.search_for("________"))  # for horizontal lines

    # 3. Check for Dates
    date_patterns = [
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*[\s\-.,]*\d{4}",
        r"\d{2}/\d{4}",
        r"\d{4}",
    ]
    contains_date = any(re.search(p, full_text, re.IGNORECASE) for p in date_patterns)
    results["contains_dates"] = contains_date

    # 4. Long Bullet Points
    bullet_lines = [line.strip() for line in full_text.splitlines() if line.strip().startswith(("•", "-", "*"))]
    long_bullets = [line for line in bullet_lines if len(line.split()) > 20]
    results["long_bullet_points_count"] = len(long_bullets)

    # 5. Design Check: Fonts and Sizes
    standard_fonts = {
        "Times-Roman", "Times New Roman", "Helvetica", "Arial", "Calibri",
        "Cambria", "Georgia", "Garamond", "Verdana", "Roboto", "Lato", "Open Sans"
    }
    used_fonts = {font_name for font_name, _ in fonts}
    non_standard_fonts = used_fonts - standard_fonts
    invalid_sizes = [round(size) for _, size in fonts if round(size) < 10 or round(size) > 12]
    all_sizes_ok = len(invalid_sizes) == 0

    results["standard_fonts_used"] = list(used_fonts & standard_fonts)
    results["non_standard_fonts_used"] = list(non_standard_fonts)
    results["font_sizes_outside_10_12"] = invalid_sizes
    results["images_in_pdf"] = images
    results["horizontal_lines"] = lines

    # --- Score Calculation ---
    score = 0
    score_breakdown = {}

    # File type & size
    file_score = 8 if results["file_type"].lower() == ".pdf" else 0
    size_score = 7 if results["file_size_mb"] <= 2 else 5
    score_breakdown["File Type/Size"] = file_score + size_score
    score += score_breakdown["File Type/Size"]

    # Dates
    score_breakdown["Dates Present"] = 15 if results["contains_dates"] else 0
    score += score_breakdown["Dates Present"]

    # Bullet length
    if results["long_bullet_points_count"] <= 2:
        bullet_score = 20
    elif results["long_bullet_points_count"] <= 5:
        bullet_score = 10
    else:
        bullet_score = 5
    score_breakdown["Bullet Point Length"] = bullet_score
    score += bullet_score

    # Font standards
    non_std_fonts = len(results["non_standard_fonts_used"])
    if non_std_fonts == 0:
        font_score = 25
    elif non_std_fonts <= 2:
        font_score = 10
    else:
        font_score = 0
    score_breakdown["Font Standards"] = font_score
    score += font_score

    # Font size
    invalid_sizes = len(results["font_sizes_outside_10_12"])
    if invalid_sizes == 0:
        size_score = 15
    elif invalid_sizes <= 2:
        size_score = 7
    else:
        size_score = 0
    score_breakdown["Font Size Range"] = size_score
    score += size_score

    # Design (images/lines)
    if results["images_in_pdf"] <= 2 and results["horizontal_lines"] <= 3:
        design_score = 10
    else:
        design_score = 5
    score_breakdown["Design Cleanliness"] = design_score
    score += design_score

    results["final_resume_score"] = score
    results["score_breakdown"] = score_breakdown

    print("\n📋 Resume Analysis Results:")
    for k, v in results.items():
        if k not in ["score_breakdown"]:
            print(f"{k.replace('_', ' ').title()}: {v}")

    print("\n📊 Score Breakdown:")
    for k, v in results["score_breakdown"].items():
        max_score = (
            "25" if "Font Standards" in k else
            "15" if "Date" in k or "Size" in k or "File" in k else
            "20" if "Bullet" in k else
            "10"
        )
        print(f"{k}: {v}/{max_score}")
    print(f"\n⭐ External parameters score: {results['final_resume_score']} / 100")

    return results['final_resume_score']


# # --- Main Execution ---
# if __name__ == "__main__":
#     file_path = input("📂 Enter the full path to the resume PDF: ").strip()
#     if not os.path.exists(file_path):
#         print("❌ File not found. Please check the path.")
#     else:
#         results = analyze_resume(file_path)

#         print("\n📋 Resume Analysis Results:")
#         for k, v in results.items():
#             if k not in ["score_breakdown"]:
#                 print(f"{k.replace('_', ' ').title()}: {v}")

#         print("\n📊 Score Breakdown:")
#         for k, v in results["score_breakdown"].items():
#             max_score = (
#                 "25" if "Font Standards" in k else
#                 "15" if "Date" in k or "Size" in k or "File" in k else
#                 "20" if "Bullet" in k else
#                 "10"
#             )
#             print(f"{k}: {v}/{max_score}")

#         print(f"\n⭐ Final Resume Score: {results['final_resume_score']} / 100")
