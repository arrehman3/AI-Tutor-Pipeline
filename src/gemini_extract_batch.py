import argparse
import json
import re
import time
from pathlib import Path

import pymupdf
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# PATHS AND CONFIGURATION
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

PDF_DIR = BASE_DIR / "PDFS"
OUTPUT_DIR = BASE_DIR / "output" / "llm"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "gemini-3.1-flash-lite"

load_dotenv(BASE_DIR / ".env")


# ---------------------------------------------------------
# PYDANTIC EXTRACTION SCHEMA
# ---------------------------------------------------------


class Option(BaseModel):
    label: str
    text: str
    pdf_page: int


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
    text: str = ""
    pdf_page: int
    options: list[Option] = Field(default_factory=list)
    parts: list[Part] = Field(default_factory=list)


class PaperExtraction(BaseModel):
    source_file: str
    subject_code: str
    component: str
    questions: list[Question]
    extraction_notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------
# GEMINI PROMPT
# ---------------------------------------------------------

EXTRACTION_PROMPT = """
You are an expert Cambridge International examination-paper
structure extraction system.

Your task is to extract the complete question hierarchy from
the supplied Cambridge question paper PDF.

Return structured JSON matching the supplied schema.

IMPORTANT: This is extraction, NOT question solving.

Do not answer questions.
Do not calculate answers.
Do not invent missing information.
Preserve the original wording as accurately as possible.

---------------------------------------------------------
1. QUESTION HIERARCHY
---------------------------------------------------------

Extract the following hierarchy:

Paper
  Question
    Part
      Subpart

Example:

1
  (a)
  (b)
  (c)
    (i)
    (ii)
  (d)
    (i)
    (ii)

Rules:

- A numbered question such as 1, 2, 3 is a Question.
- Labels such as (a), (b), (c) are Parts.
- Labels such as (i), (ii), (iii) are Subparts.
- Do not interpret printed page numbers as question numbers.
- Do not interpret graph labels, chemical numbers, table numbers,
  or diagram labels as questions or parts.
- Extract only actual examination questions.

---------------------------------------------------------
2. PART AND SUBPART TEXT
---------------------------------------------------------

This rule is especially important.

If a part contains subparts, keep the shared introductory
wording at the parent Part level.

Put the individual wording of each subpart at its own level.

Example source:

(d) Explain the reaction.

    (i) Describe the first step. [2]

    (ii) Explain the final product. [3]

Expected structure:

Part:
    label: "(d)"
    text: "Explain the reaction."
    marks: null
    subparts:
        - label: "(i)"
          text: "Describe the first step."
          marks: 2

        - label: "(ii)"
          text: "Explain the final product."
          marks: 3

Do NOT duplicate subpart wording in the parent Part text.

If the part has no introductory wording and consists only of
subparts, use an empty string for the parent Part text.

Do not move subpart wording into the parent Part.

---------------------------------------------------------
3. MARKS
---------------------------------------------------------

Extract marks exactly as printed.

Examples:

[2] -> marks: 2
[3] -> marks: 3
[1] -> marks: 1

Rules:

- Marks belong to the exact question, part, or subpart
  where they are printed.
- If marks are printed beside subparts, assign them to
  the subparts, not the parent part.
- If a parent part has subparts and no separate parent marks,
  use null for parent marks.
- Never invent marks.
- If marks cannot be determined, use null.

---------------------------------------------------------
4. PAGE NUMBERS
---------------------------------------------------------

Use the actual PDF page number, starting from page 1.

Do not confuse printed page numbers with PDF page numbers.

Every question, part, and subpart must have its own
correct PDF page number.

If a question continues onto another page, assign the page
where the corresponding part or subpart actually begins.

---------------------------------------------------------
5. QUESTION INTRODUCTORY TEXT
---------------------------------------------------------

If text appears between the question number and its first
part, preserve it in introductory_text.

Do not duplicate the same text in the first part.

If there is no introductory text, use an empty string.

---------------------------------------------------------
6. COMPLETENESS
---------------------------------------------------------

Extract EVERY question from the entire PDF.

Do not stop after the first few questions.

Do not omit questions at the end of the paper.

Check the final pages before completing the extraction.

Do not merge separate questions.

Do not split one question into multiple questions.

---------------------------------------------------------
7. EXTRACTION NOTES
---------------------------------------------------------

Only report important, distinct uncertainties.

Do not repeat the same note.

Do not include routine observations or explanations
of your extraction process.

Maximum 10 concise notes.

If there are no important uncertainties, return an empty list.


---------------------------------------------------------
MULTIPLE-CHOICE QUESTIONS
---------------------------------------------------------

Some papers contain multiple-choice questions.

For each multiple-choice question:

1. Extract the question stem into the Question.text field.
2. Extract each answer choice into the Question.options array.
3. Preserve the original option labels: A, B, C, D.
4. Preserve the complete wording of each option.
5. Record the PDF page number for each option.
6. Do not store answer choices as Parts or Subparts.
7. Do not omit answer choices even if they contain diagrams,
   symbols, equations or chemical structures.
8. Do not identify the correct answer or solve the question.

Example:

Question:
    number: "1"
    text: "Which of the following is a base unit?"
    options:
        - label: "A"
          text: "metre"
        - label: "B"
          text: "percentage uncertainty"
        - label: "C"
          text: "quark flavour"
        - label: "D"
          text: "spring constant"

For multiple-choice questions:
- parts must be an empty list.
- options must contain the answer choices.

For structured questions:
- options must be an empty list.
- use parts and subparts as appropriate.

A question may contain introductory text and a separate
question stem. Preserve both when they are distinct.
---------------------------------------------------------


---------------------------------------------------------
8. OUTPUT
---------------------------------------------------------

Return only structured data matching the supplied schema.

Preserve the original question numbering and labels.

Use empty strings where text is genuinely absent.

Use null where marks are genuinely unavailable.

Do not add extra fields.
"""


# ---------------------------------------------------------
# FILE METADATA
# ---------------------------------------------------------


def get_paper_metadata(pdf_path: Path) -> dict:
    """
    Extract subject, session, document type and component
    from filenames such as:

    9701_s22_qp_41.pdf
    """

    pattern = (
        r"^(?P<subject>\d{4})_"
        r"(?P<session>[a-z]\d{2})_"
        r"(?P<document>qp|ms)_"
        r"(?P<component>\d{2})\.pdf$"
    )

    match = re.match(pattern, pdf_path.name, re.IGNORECASE)

    if not match:
        raise ValueError(
            f"Filename does not match expected format: " f"{pdf_path.name}"
        )

    return {
        "subject_code": match.group("subject"),
        "session_code": match.group("session"),
        "document_type": match.group("document").lower(),
        "component": match.group("component"),
    }


# ---------------------------------------------------------
# PDF INFORMATION
# ---------------------------------------------------------


def get_pdf_page_count(pdf_path: Path) -> int:
    """Return the number of pages in the source PDF."""

    with pymupdf.open(pdf_path) as document:
        return len(document)


# ---------------------------------------------------------
# RESPONSE PARSING
# ---------------------------------------------------------


def parse_gemini_response(response) -> PaperExtraction:
    """
    Prefer Gemini's parsed response.

    If unavailable, parse the raw response text as JSON.
    Then validate the result against the Pydantic schema.
    """

    if response.parsed is not None:
        raw_data = response.parsed

    else:
        response_text = response.text

        if not response_text:
            raise RuntimeError("Gemini returned an empty response.")

        try:
            raw_data = json.loads(response_text)

        except json.JSONDecodeError as error:
            raise RuntimeError(
                "Gemini response was not complete valid JSON. "
                "The response may have been truncated or malformed.\n"
                f"JSON error: {error}\n"
                f"Response length: {len(response_text)} characters\n"
                f"Response ending: {response_text[-1000:]}"
            ) from error

    return PaperExtraction.model_validate(raw_data)


# ---------------------------------------------------------
# SINGLE PAPER EXTRACTION
# ---------------------------------------------------------


def extract_paper(client, pdf_path: Path) -> dict:

    metadata = get_paper_metadata(pdf_path)

    output_path = OUTPUT_DIR / f"{pdf_path.stem}_gemini.json"

    print("\n" + "=" * 65)
    print(f"Processing: {pdf_path.name}")
    print("=" * 65)

    uploaded_file = None

    try:

        # Upload original PDF to Gemini
        print("Uploading PDF...")

        uploaded_file = client.files.upload(file=str(pdf_path))

        print("PDF uploaded successfully.")

        # Generate structured extraction
        print(f"Extracting with {MODEL}...")

        response = client.models.generate_content(
            model=MODEL,
            contents=[
                uploaded_file,
                EXTRACTION_PROMPT,
            ],
            config={
                "response_mime_type": "application/json",
                "response_schema": PaperExtraction,
                "temperature": 0,
                "max_output_tokens": 12000,
            },
        )

        # Parse and validate response
        extraction = parse_gemini_response(response)

        # Filename is authoritative for metadata
        extraction = extraction.model_copy(
            update={
                "source_file": pdf_path.name,
                "subject_code": metadata["subject_code"],
                "component": metadata["component"],
            }
        )

        # Save extraction
        output_path.write_text(
            extraction.model_dump_json(indent=2),
            encoding="utf-8",
        )

        # Summary
        question_count = len(extraction.questions)

        part_count = sum(len(question.parts) for question in extraction.questions)

        subpart_count = sum(
            len(part.subparts)
            for question in extraction.questions
            for part in question.parts
        )

        page_count = get_pdf_page_count(pdf_path)

        print("\nExtraction successful.")
        print(f"PDF pages: {page_count}")
        print(f"Questions: {question_count}")
        print(f"Parts: {part_count}")
        print(f"Subparts: {subpart_count}")
        print(f"Extraction notes: {len(extraction.extraction_notes)}")
        print(f"Saved: {output_path}")

        return {
            "file": pdf_path.name,
            "status": "success",
            "questions": question_count,
            "parts": part_count,
            "subparts": subpart_count,
            "output": str(output_path.relative_to(BASE_DIR)),
        }

    finally:

        # Delete temporary Gemini upload
        if uploaded_file is not None:

            try:
                client.files.delete(name=uploaded_file.name)
                print("Temporary Gemini upload deleted.")

            except Exception as error:
                print("Warning: Could not delete temporary " f"Gemini upload: {error}")


# ---------------------------------------------------------
# BATCH PROCESSING
# ---------------------------------------------------------


def main():

    parser = argparse.ArgumentParser(
        description="Batch extract Cambridge question papers using Gemini."
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess papers even if output JSON already exists.",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=5.0,
        help="Delay in seconds between API requests (default: 5).",
    )

    args = parser.parse_args()

    # Discover question papers only
    pdf_files = sorted(PDF_DIR.glob("*/*_qp_*.pdf"))

    if not pdf_files:
        print(f"No question paper PDFs found in {PDF_DIR}")
        return

    # Check filenames before making API calls
    valid_files = []

    for pdf_path in pdf_files:

        try:
            get_paper_metadata(pdf_path)
            valid_files.append(pdf_path)

        except ValueError as error:
            print(f"Skipping invalid filename: {error}")

    print("\nCAMBRIDGE QUESTION PAPER BATCH EXTRACTION")
    print("-" * 65)
    print(f"PDF directory: {PDF_DIR}")
    print(f"Discovered papers: {len(valid_files)}")
    print(f"Model: {MODEL}")
    print(f"Overwrite existing: {args.overwrite}")

    # API client
    client = genai.Client()

    results = []

    try:

        for index, pdf_path in enumerate(valid_files, start=1):

            output_path = OUTPUT_DIR / f"{pdf_path.stem}_gemini.json"

            print(f"\nPaper {index}/{len(valid_files)}")

            # Skip existing outputs unless overwrite enabled
            if output_path.exists() and not args.overwrite:

                print(f"Skipping existing output: {output_path.name}")

                results.append(
                    {
                        "file": pdf_path.name,
                        "status": "skipped",
                        "reason": "Output already exists",
                        "output": str(output_path.relative_to(BASE_DIR)),
                    }
                )

                continue

            try:

                result = extract_paper(
                    client,
                    pdf_path,
                )

                results.append(result)

            except Exception as error:

                print(f"\nFAILED: {pdf_path.name}")
                print(f"Reason: {error}")

                results.append(
                    {
                        "file": pdf_path.name,
                        "status": "failed",
                        "error": str(error),
                    }
                )

            # Avoid sending requests too quickly
            if index < len(valid_files) and args.delay > 0:
                time.sleep(args.delay)

    finally:
        client.close()

    # -----------------------------------------------------
    # SAVE BATCH REPORT
    # -----------------------------------------------------

    success_count = sum(result["status"] == "success" for result in results)

    skipped_count = sum(result["status"] == "skipped" for result in results)

    failed_count = sum(result["status"] == "failed" for result in results)

    report = {
        "model": MODEL,
        "total_discovered": len(valid_files),
        "successful": success_count,
        "skipped": skipped_count,
        "failed": failed_count,
        "results": results,
    }

    report_path = OUTPUT_DIR / "batch_extraction_report.json"

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    # -----------------------------------------------------
    # FINAL SUMMARY
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("BATCH EXTRACTION SUMMARY")
    print("=" * 65)

    print(f"Discovered: {len(valid_files)}")
    print(f"Successful: {success_count}")
    print(f"Skipped:    {skipped_count}")
    print(f"Failed:     {failed_count}")

    print(f"\nReport saved: {report_path}")

    if failed_count:
        print("\nSome papers failed. Check the batch report.")
    else:
        print("\nBatch processing completed without failures.")


if __name__ == "__main__":
    main()
