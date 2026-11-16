
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXAM_FILE = (
    PROJECT_ROOT
    / "output/domain/exam/paper_9701_s22_qp_41.json"
)

EXTRACTION_FILE = (
    PROJECT_ROOT
    / "output/llm/9701_s22_qp_41_gemini.json"
)


def validate():
    domain = json.loads(EXAM_FILE.read_text(encoding="utf-8"))
    extraction = json.loads(
        EXTRACTION_FILE.read_text(encoding="utf-8")
    )

    errors = []
    ids = set()

    def check_id(object_id, location):
        if not object_id:
            errors.append(f"Missing ID: {location}")
        elif object_id in ids:
            errors.append(f"Duplicate ID: {object_id}")
        else:
            ids.add(object_id)

    check_id(domain["paper_id"], "Paper")

    if domain["source_file"] != extraction["source_file"]:
        errors.append("Source filename does not match extraction")

    if len(domain["questions"]) != len(extraction["questions"]):
        errors.append("Question count differs from extraction")

    part_count = 0
    subpart_count = 0

    for question in domain["questions"]:
        check_id(question["question_id"], "Question")

        if question["pdf_page"] < 1:
            errors.append(
                f"Invalid page for question {question['number']}"
            )

        for part in question["parts"]:
            part_count += 1
            check_id(part["part_id"], "Part")

            expected_part_id = (
                f"{question['question_id']}_{part['label']}"
            )
            if part["part_id"] != expected_part_id:
                errors.append(
                    f"Incorrect part ID: {part['part_id']}"
                )

            for subpart in part["subparts"]:
                subpart_count += 1
                check_id(subpart["subpart_id"], "Subpart")

                expected_subpart_id = (
                    f"{part['part_id']}_{subpart['label']}"
                )
                if subpart["subpart_id"] != expected_subpart_id:
                    errors.append(
                        f"Incorrect subpart ID: "
                        f"{subpart['subpart_id']}"
                    )

    print("EXAM DOMAIN VALIDATION")
    print("----------------------")
    print(f"Paper: {domain['paper_id']}")
    print(f"Questions: {len(domain['questions'])}")
    print(f"Parts: {part_count}")
    print(f"Subparts: {subpart_count}")
    print(f"Unique IDs: {len(ids)}")

    if errors:
        print(f"\nFAILED: {len(errors)} error(s)")
        for error in errors:
            print("-", error)
    else:
        print("\nPASSED: No structural errors found.")


if __name__ == "__main__":
    validate()
