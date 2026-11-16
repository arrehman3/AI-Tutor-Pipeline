
import json
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent.parent

LLM_DIR = BASE_DIR / "output" / "llm"
EXTRACTED_DIR = BASE_DIR / "output" / "extracted"
REPORT_DIR = BASE_DIR / "output" / "validation"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def find_source_document(source_file):
    """Find the corresponding PyMuPDF extraction JSON."""
    matches = list(
        EXTRACTED_DIR.rglob(Path(source_file).stem + ".json")
    )

    return load_json(matches[0]) if matches else None


def validate_paper(paper, source_document=None):
    errors = []
    warnings = []

    questions = paper.get("questions", [])

    if not questions:
        errors.append("No questions extracted.")

    question_numbers = [
        str(q.get("number", ""))
        for q in questions
    ]

    duplicates = [
        number
        for number, count in Counter(question_numbers).items()
        if count > 1
    ]

    for number in duplicates:
        errors.append(f"Duplicate question number: {number}")

    total_pages = (
        source_document.get("total_pages")
        if source_document else None
    )

    def check_page(page, location):
        if not isinstance(page, int):
            errors.append(f"{location}: invalid or missing page number.")
        elif page < 1:
            errors.append(f"{location}: page number must be positive.")
        elif total_pages and page > total_pages:
            errors.append(
                f"{location}: page {page} exceeds "
                f"PDF page count ({total_pages})."
            )

    for question in questions:
        q_number = str(question.get("number", ""))
        q_location = f"Question {q_number}"

        check_page(question.get("pdf_page"), q_location)

        parts = question.get("parts", [])
        part_labels = [str(p.get("label", "")) for p in parts]

        for label, count in Counter(part_labels).items():
            if count > 1:
                errors.append(
                    f"{q_location}: duplicate part label ({label})."
                )

        for part in parts:
            label = str(part.get("label", ""))
            location = f"{q_location}, Part ({label})"

            if not label:
                errors.append(f"{location}: missing label.")

            part_text = part.get("text", "")
            subparts = part.get("subparts", [])

            # Empty parent text is valid when it contains subparts.
            if not str(part_text).strip() and not subparts:
                warnings.append(f"{location}: empty question text.")

            check_page(part.get("pdf_page"), location)

            marks = part.get("marks")

            if marks is not None and (
                not isinstance(marks, int) or marks < 0
            ):
                errors.append(f"{location}: invalid marks value.")

            subpart_labels = [
                str(s.get("label", ""))
                for s in subparts
            ]

            for sublabel, count in Counter(subpart_labels).items():
                if count > 1:
                    errors.append(
                        f"{location}: duplicate subpart label "
                        f"({sublabel})."
                    )

            for subpart in subparts:
                sublabel = str(subpart.get("label", ""))
                sub_location = f"{location}, Subpart ({sublabel})"

                if not sublabel:
                    errors.append(f"{sub_location}: missing label.")

                if not str(subpart.get("text", "")).strip():
                    warnings.append(
                        f"{sub_location}: empty question text."
                    )

                check_page(subpart.get("pdf_page"), sub_location)

                sub_marks = subpart.get("marks")

                if sub_marks is not None and (
                    not isinstance(sub_marks, int) or sub_marks < 0
                ):
                    errors.append(
                        f"{sub_location}: invalid marks value."
                    )

                if sub_marks is None:
                    warnings.append(
                        f"{sub_location}: marks not extracted."
                    )

            if marks is None and not subparts:
                warnings.append(
                    f"{location}: marks not extracted."
                )

    return {
        "source_file": paper.get("source_file"),
        "question_count": len(questions),
        "part_count": sum(
            len(q.get("parts", [])) for q in questions
        ),
        "subpart_count": sum(
            len(part.get("subparts", []))
            for q in questions
            for part in q.get("parts", [])
        ),
        "errors": errors,
        "warnings": warnings,
        "status": "PASS" if not errors else "FAIL"
    }


def main():
    files = sorted(LLM_DIR.glob("*_gemini.json"))

    if not files:
        raise FileNotFoundError(
            f"No Gemini extraction files found in {LLM_DIR}"
        )

    reports = []

    for path in files:
        paper = load_json(path)

        source_document = find_source_document(
            paper["source_file"]
        )

        report = validate_paper(paper, source_document)
        reports.append(report)

        print(f"\n{report['source_file']}")
        print(f"Status: {report['status']}")
        print(f"Questions: {report['question_count']}")
        print(f"Parts: {report['part_count']}")
        print(f"Subparts: {report['subpart_count']}")
        print(f"Errors: {len(report['errors'])}")
        print(f"Warnings: {len(report['warnings'])}")

        for error in report["errors"]:
            print(f"  ERROR: {error}")

        for warning in report["warnings"]:
            print(f"  WARNING: {warning}")

    output_path = REPORT_DIR / "validation_report.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(reports, file, indent=2, ensure_ascii=False)

    print(f"\nValidation report saved: {output_path}")


if __name__ == "__main__":
    main()
