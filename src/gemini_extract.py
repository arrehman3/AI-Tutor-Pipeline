
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = BASE_DIR / "PDFS" / "Chem" / "9701_s22_qp_41.pdf"

OUTPUT_DIR = BASE_DIR / "output" / "llm"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "9701_s22_qp_41_gemini.json"

MODEL = "gemini-3.1-flash-lite"


# --------------------------------------------------
# Structured output schema
# --------------------------------------------------

class Subpart(BaseModel):
    label: str
    text: str
    marks: int | None = None
    pdf_page: int


class Part(BaseModel):
    label: str
    text: str
    marks: int | None = None
    pdf_page: int
    subparts: list[Subpart] = Field(default_factory=list)


class Question(BaseModel):
    number: str
    introductory_text: str = ""
    pdf_page: int
    parts: list[Part] = Field(default_factory=list)


class PaperExtraction(BaseModel):
    source_file: str
    subject_code: str
    component: str
    questions: list[Question]
    extraction_notes: list[str] = Field(default_factory=list)


# --------------------------------------------------
# Instructions for Gemini
# --------------------------------------------------

PROMPT = """
You are extracting a Cambridge International A-Level Chemistry
question paper into a hierarchical JSON structure.

Read the entire PDF, including its visual layout.

HIERARCHY RULES:

1. A numbered heading such as Question 1, Question 2 is a
   top-level Question.

2. Labels (a), (b), (c), etc. are Parts belonging to the
   current Question.

3. Labels (i), (ii), (iii), etc. are Subparts belonging
   to the immediately preceding Part.

4. A Part can contain:
   - Its own question text, without subparts.
   - Introductory text followed by Subparts.
   - Subparts without any introductory text.

5. CRITICAL: If a line begins with (i), (ii), (iii), etc.,
   its question wording MUST be stored in a Subpart object.
   Do not store that wording as the Part's own text.

6. If a Part contains Subparts, its text field must contain
   ONLY introductory or shared text that applies to those
   Subparts. If there is no introductory text, use an empty
   string.

7. Never omit a Subpart just because its wording starts on
   the same line as its label.

8. Preserve the exact original wording as closely as possible.

9. Extract marks from the corresponding question or subpart.
   Do not assign the sum of child marks to the parent Part.

10. If marks are printed only against Subparts, the parent
    Part's marks must be null.

11. Use 1-based PDF page numbers.

12. A question or part may continue across pages. Maintain
    the correct hierarchy.

13. Do not mistake page numbers, graph labels, chemical
    formula numbers, or table entries for question numbers.

14. Do not solve questions or invent missing text or marks.

15. Record unclear text, diagrams, and extraction uncertainties
    in extraction_notes.

IMPORTANT EXAMPLE:

If the PDF contains:

(d) (i) Calculate the standard enthalpy change. [3]
    (ii) Suggest how the lattice energy differs. [2]

The JSON must contain:

Part:
{
    "label": "d",
    "text": "",
    "marks": null,
    "subparts": [
        {
            "label": "i",
            "text": "Calculate the standard enthalpy change.",
            "marks": 3
        },
        {
            "label": "ii",
            "text": "Suggest how the lattice energy differs.",
            "marks": 2
        }
    ]
}

Return the complete paper structure.
"""


def main():
    load_dotenv(BASE_DIR / ".env")

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY was not found in .env")

    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    client = genai.Client(api_key=api_key)

    uploaded_file = None

    try:
        print(f"Uploading: {PDF_PATH.name}")

        uploaded_file = client.files.upload(
            file=str(PDF_PATH)
        )

        print(f"Uploaded successfully. Model: {MODEL}")
        print("Gemini is analysing the paper...")

        response = client.models.generate_content(
            model=MODEL,
            contents=[
                PROMPT,
                uploaded_file
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PaperExtraction,
                temperature=0,
                max_output_tokens=12000
            )
        )

        if response.parsed is None:
            raise RuntimeError(
                f"Gemini returned no parsed result.\n{response.text}"
            )

        extraction = PaperExtraction.model_validate(
            response.parsed
        )

        # Save validated structured output
        with OUTPUT_PATH.open("w", encoding="utf-8") as f:
            json.dump(
                extraction.model_dump(),
                f,
                ensure_ascii=False,
                indent=2
            )

        print("\nExtraction completed.")
        print(f"Questions found: {len(extraction.questions)}")
        print(f"Output saved: {OUTPUT_PATH}")

        for question in extraction.questions:
            print(
                f"Question {question.number}: "
                f"{len(question.parts)} parts"
            )

        if extraction.extraction_notes:
            print("\nExtraction notes:")
            for note in extraction.extraction_notes:
                print(f"- {note}")

    finally:
        # Uploaded Gemini files are temporary.
        if uploaded_file is not None:
            try:
                client.files.delete(name=uploaded_file.name)
                print("Temporary uploaded file deleted.")
            except Exception as error:
                print(f"Could not delete temporary file: {error}")

        client.close()


if __name__ == "__main__":
    main()
