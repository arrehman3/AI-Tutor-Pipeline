
from pathlib import Path
import json
import re

BASE_DIR = Path(__file__).resolve().parent.parent
EXTRACTED_DIR = BASE_DIR / "output" / "extracted"
OUTPUT_DIR = BASE_DIR / "output" / "analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cambridge question and part patterns
question_pattern = re.compile(
    r"^\s*(\d{1,2})(?:\s+\(([a-z])\))?(?=\s|$)",
    re.IGNORECASE
)

part_pattern = re.compile(r"^\s*\(([a-z])\)", re.IGNORECASE)

# Common Roman numeral subparts
subpart_pattern = re.compile(
    r"^\s*\((i{1,3}|iv|v|vi{0,3}|ix|x)\)",
    re.IGNORECASE
)

marks_pattern = re.compile(r"\[(\d+)\]")

# Exclude common examination duration lines
duration_pattern = re.compile(
    r"^\s*\d+(?:\.\d+)?\s*(?:hours?|hrs?|minutes?|mins?)\b",
    re.IGNORECASE
)

results = []

for json_path in sorted(EXTRACTED_DIR.rglob("*_qp_*.json")):

    with json_path.open(encoding="utf-8") as file:
        document = json.load(file)

    detected_pages = []
    active_part = False

    for page in document["pages"]:

        lines = page["text"].splitlines()
        events = []

        for line_number, original_line in enumerate(lines, start=1):

            line = original_line.strip()

            if not line:
                continue

            event = {
                "line_number": line_number,
                "text": line
            }

            # Ignore examination duration and similar header text
            if duration_pattern.match(line):
                continue

            question_match = question_pattern.match(line)
            part_match = part_pattern.match(line)
            subpart_match = subpart_pattern.match(line)

            if question_match:

                event["type"] = "question_candidate"
                event["question_number"] = question_match.group(1)

                active_part = False

                if question_match.group(2):
                    event["part"] = question_match.group(2)
                    active_part = True

            elif part_match:

                event["type"] = "part_candidate"
                event["part"] = part_match.group(1)
                active_part = True

                # Check whether the same line also contains a subpart.
                remainder = line[part_match.end():].strip()
                nested_match = subpart_pattern.match(remainder)

                if nested_match:
                    event["subpart"] = nested_match.group(1)
                    event["contains_subpart"] = True

            elif subpart_match and active_part:

                event["type"] = "subpart_candidate"
                event["subpart"] = subpart_match.group(1)

            else:
                event["type"] = "content"

            marks = marks_pattern.findall(line)

            if marks:
                event["possible_marks"] = [int(m) for m in marks]

            if event["type"] != "content" or marks:
                events.append(event)

        detected_pages.append({
            "page_number": page["page_number"],
            "events": events
        })

    results.append({
        "source_file": document["source_file"],
        "total_pages": document["total_pages"],
        "pages": detected_pages
    })

output_file = OUTPUT_DIR / "structural_candidates.json"

with output_file.open("w", encoding="utf-8") as file:
    json.dump(results, file, ensure_ascii=False, indent=2)

print(f"Question papers analysed: {len(results)}")

for document in results:
    questions = sum(
        event["type"] == "question_candidate"
        for page in document["pages"]
        for event in page["events"]
    )

    parts = sum(
        event["type"] == "part_candidate"
        for page in document["pages"]
        for event in page["events"]
    )

    subparts = sum(
        event["type"] == "subpart_candidate"
        or event.get("contains_subpart", False)
        for page in document["pages"]
        for event in page["events"]
    )

    print(
        f"{document['source_file']}: "
        f"{questions} question candidates, "
        f"{parts} part candidates, "
        f"{subparts} subpart candidates"
    )

print(f"\nSaved: {output_file}")
