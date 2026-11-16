
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "output/analysis/structural_candidates.json"
OUTPUT_DIR = BASE_DIR / "output/analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with INPUT_FILE.open(encoding="utf-8") as file:
    documents = json.load(file)

papers = []

for document in documents:

    if "_qp_" not in document["source_file"]:
        continue

    paper = {
        "source_file": document["source_file"],
        "questions": []
    }

    current_question = None
    current_part = None

    for page in document["pages"]:

        for event in page["events"]:

            event_type = event["type"]
            text = event["text"]

            if event_type == "question_candidate":

                current_question = {
                    "number": event["question_number"],
                    "parts": [],
                    "page_started": page["page_number"]
                }

                paper["questions"].append(current_question)
                current_part = None

                # Some question headings also contain a part, e.g. 1 (a)
                if event.get("part"):
                    current_part = {
                        "label": event["part"],
                        "page": page["page_number"],
                        "raw_text": [text],
                        "mark_candidates": []
                    }
                    current_question["parts"].append(current_part)

            elif event_type == "part_candidate":

                if current_question is not None:
                    current_part = {
                        "label": event["part"],
                        "page": page["page_number"],
                        "raw_text": [text],
                        "mark_candidates": []
                    }

                    current_question["parts"].append(current_part)

            elif current_part is not None:

                current_part["raw_text"].append(text)

                if event.get("possible_marks"):
                    current_part["mark_candidates"].extend(
                        event["possible_marks"]
                    )

    papers.append(paper)

output_file = OUTPUT_DIR / "exam_candidates.json"

with output_file.open("w", encoding="utf-8") as file:
    json.dump(papers, file, ensure_ascii=False, indent=2)

print(f"Papers processed: {len(papers)}")

for paper in papers:
    question_count = len(paper["questions"])
    part_count = sum(
        len(q["parts"]) for q in paper["questions"]
    )

    print(
        f"{paper['source_file']}: "
        f"{question_count} questions, "
        f"{part_count} parts"
    )

print(f"\nSaved: {output_file}")
