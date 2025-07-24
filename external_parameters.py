import fitz  # PyMuPDF

def analyze_resume(pdf_path, bullet_threshold=15, words_threshold_min=500, words_threshold_max=600):
    doc = fitz.open(pdf_path)
    fonts_used = set()
    total_lines = 0
    long_bullets = 0
    total_words = 0

    # Preferred resume-friendly fonts
    preferred_fonts = {
        "Times-Roman", "Times New Roman", "Helvetica", "Arial", "Calibri",
        "Cambria", "Georgia", "Garamond", "Verdana", "Roboto", "Lato", "Open Sans"
    }

    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    text_line = " ".join([span["text"].strip() for span in line["spans"] if span["text"].strip()])
                    if text_line.strip():
                        total_lines += 1
                        words = text_line.strip().split()
                        total_words += len(words)

                        # Count long bullet points
                        if text_line.strip().startswith(("-", "•", "*")):
                            if len(words) > bullet_threshold:
                                long_bullets += 1

                    # Collect fonts
                    for span in line["spans"]:
                        fonts_used.add(span["font"])

    doc.close()

    # ------------------ Scoring ------------------
    # Font Score (out of 30)
    font_score = 30 if any(font in preferred_fonts for font in fonts_used) else 10

    # Bullet Points Score (out of 30)
    if total_lines > 0:
        bullet_ratio = long_bullets / total_lines
    else:
        bullet_ratio = 0
    bullet_score = 30 if bullet_ratio <= 1/3 else int(30 * (1 - bullet_ratio))  # reduce if too many long bullets

    # Word Count Score (out of 40)
    if words_threshold_min <= total_words <= words_threshold_max:
        word_score = 40
    elif total_words < words_threshold_min:
        word_score = max(10, int(40 * (total_words / words_threshold_min)))  # scale up proportionally
    else:  # too many words
        word_score = max(10, int(40 * (words_threshold_max / total_words)))

    total_score = font_score + bullet_score + word_score  # Max 100

    # ------------------ Output ------------------
    print("=== Resume Analysis ===")
    print(f"Total lines: {total_lines}")
    print(f"Total words: {total_words}")
    print(f"Fonts used: {', '.join(sorted(fonts_used))}")
    print(f"Long bullet points (> {bullet_threshold} words): {long_bullets}")
    print("\n--- Scoring Breakdown ---")
    print(f"Font Style Score (30): {font_score}")
    print(f"Bullet Points Score (30): {bullet_score}")
    print(f"Word Count Score (40): {word_score}")
    print(f"Total Resume Quality Score: {total_score}/100")
    print("--------------------------")

# Usage
# analyze_resume("your_resume.pdf")
