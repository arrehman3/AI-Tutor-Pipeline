
import json
import re
from pathlib import Path

from exam_models import Paper


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "output/llm/9701_s22_qp_41_gemini.json"
)

OUTPUT_DIR = PROJECT_ROOT / "output/domain/exam"


def build_paper(data: dict) -> Paper:
    source_file = data["source_file"]

    # Extract session and paper type from filenames such as:
    # 9701_s22_qp_41.pdf
    match = re.fullmatch(
        r"(?P<subject>\d{4})_"
        r"(?P<session>[a-z]\d{2})_"
        r"(?P<document_type>qp|ms)_"
        r"(?P<component>\d{2})\.pdf",
        source_file,
    )

    if not match:
        raise ValueError(f"Unexpected source filename: {source_file}")

    session_code = match.group("session")
    document_type = match.group("document_type")

    paper_id = (
        f"paper_{data['subject_code']}_"
        f"{session_code}_{document_type}_{data['component']}"
    )

    questions = []

    for question_data in data["questions"]:
        question_number = question_data["number"]
        question_id = f"{paper_id}_q{question_number}"

        parts = []

        for part_data in question_data.get("parts", []):
            part_label = part_data["label"]
            part_id = f"{question_id}_{part_label}"

            subparts = []

            for subpart_data in part_data.get("subparts", []):
                subpart_label = subpart_data["label"]

                subparts.append({
                    "subpart_id": f"{part_id}_{subpart_label}",
                    "label": subpart_label,
                    "text": subpart_data["text"],
                    "marks": subpart_data.get("marks"),
                    "pdf_page": subpart_data["pdf_page"],
                })

            parts.append({
                "part_id": part_id,
                "label": part_label,
                "text": part_data["text"],
                "marks": part_data.get("marks"),
                "pdf_page": part_data["pdf_page"],
                "subparts": subparts,
            })

        questions.append({
            "question_id": question_id,
            "number": question_number,
            "introductory_text": question_data.get(
                "introductory_text", ""
            ),
            "pdf_page": question_data["pdf_page"],
            "parts": parts,
            "topic_ids": [],
        })

    return Paper(
        paper_id=paper_id,
        source_file=source_file,
        subject_code=data["subject_code"],
        component=data["component"],
        session_code=session_code,
        document_type=document_type,
        questions=questions,
        extraction_notes=data.get("extraction_notes", []),
    )


def main():
    with INPUT_FILE.open(encoding="utf-8") as file:
        extracted_data = json.load(file)

    paper = build_paper(extracted_data)

    # Pydantic validates the complete domain object here.
    validated_data = paper.model_dump()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / f"{paper.paper_id}.json"

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(
            validated_data,
            file,
            indent=2,
            ensure_ascii=False,
        )

    question_count = len(paper.questions)
    part_count = sum(len(q.parts) for q in paper.questions)
    subpart_count = sum(
        len(part.subparts)
        for question in paper.questions
        for part in question.parts
    )

    print("Domain model created successfully.")
    print(f"Paper ID: {paper.paper_id}")
    print(f"Questions: {question_count}")
    print(f"Parts: {part_count}")
    print(f"Subparts: {subpart_count}")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()
