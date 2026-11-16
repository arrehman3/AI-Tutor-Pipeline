
import json
import os
import re
from pathlib import Path

import pymupdf
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent.parent

PDF_PATH = BASE_DIR / "PDFS" / "Chem" / "9701_s22_qp_41.pdf"

OUTPUT_DIR = BASE_DIR / "output" / "resources"
MANIFEST_DIR = BASE_DIR / "output" / "llm"

MODEL = "gemini-3.1-flash-lite"

load_dotenv(BASE_DIR / ".env")


# ---------- Structured response schema ----------

class VisualResource(BaseModel):
    resource_id: str
    type: str
    question: str
    part: str | None = None
    subpart: str | None = None
    pdf_page: int

    description: str

    # Coordinates normalized to 0–1000:
    # [x0, y0, x1, y1]
    bbox_normalized: list[int] = Field(
        description=(
            "Approximate bounding box relative to the full PDF page, "
            "in the order x0, y0, x1, y1, each from 0 to 1000."
        )
    )


class ResourceExtraction(BaseModel):
    source_file: str
    resources: list[VisualResource]
    extraction_notes: list[str] = Field(default_factory=list)


# ---------- Gemini resource identification ----------

PROMPT = """
Inspect this Cambridge Chemistry question paper and identify every
meaningful visual resource.

Include:
- Chemical structures and displayed formulae
- Reaction mechanisms
- Energy level diagrams and energy cycles
- Graphs and plotted data
- Electrochemical cells and apparatus diagrams
- Tables containing experimental or numerical data
- Chromatograms and electrophoresis results
- Blank answer boxes or spaces where a diagram or structure must be drawn

Do not include ordinary paragraphs, page numbers, headers, or footers.

For every resource:
1. Identify its question, part and subpart where possible.
2. Give the physical PDF page number, starting from 1.
3. Describe what the resource shows or what the student must complete.
4. Estimate its bounding box relative to the FULL PAGE.

Bounding box requirements:
- Use coordinates from 0 to 1000.
- Format: [x0, y0, x1, y1].
- (0, 0) is the top-left corner.
- (1000, 1000) is the bottom-right corner.
- Include enough surrounding space to preserve labels, arrows,
  axis titles and other relevant annotations.
- Do not use coordinates from a cropped region.
- If multiple resources appear separately on a page, identify them
  separately.

Use the question hierarchy visible in the paper.
Do not invent question references or resources.
Do not solve any questions.

Return only the structured response.
"""


def safe_filename(value):
    return re.sub(r"[^a-zA-Z0-9_-]", "_", value)


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(PDF_PATH)

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in .env")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    client = genai.Client(api_key=api_key)
    uploaded_file = None

    try:
        print(f"Uploading: {PDF_PATH.name}")

        uploaded_file = client.files.upload(
            file=str(PDF_PATH)
        )

        print("Gemini is identifying visual resources...")

        response = client.models.generate_content(
            model=MODEL,
            contents=[PROMPT, uploaded_file],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ResourceExtraction,
                temperature=0,
                max_output_tokens=12000,
            ),
        )

        extraction = ResourceExtraction.model_validate(
            response.parsed
        )

        # Save the original structured resource manifest.
        manifest_path = (
            MANIFEST_DIR /
            f"{PDF_PATH.stem}_resources.json"
        )

        MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

        document = pymupdf.open(PDF_PATH)

        manifest = {
            "source_file": PDF_PATH.name,
            "model": MODEL,
            "resources": [],
            "extraction_notes": extraction.extraction_notes,
            "crop_errors": [],
        }

        for resource in extraction.resources:
            page_number = resource.pdf_page

            if not 1 <= page_number <= len(document):
                manifest["crop_errors"].append(
                    f"{resource.resource_id}: invalid page "
                    f"{page_number}"
                )
                continue

            coords = resource.bbox_normalized

            if (
                len(coords) != 4
                or any(not 0 <= value <= 1000 for value in coords)
            ):
                manifest["crop_errors"].append(
                    f"{resource.resource_id}: invalid bounding box"
                )
                continue

            x0, y0, x1, y1 = coords

            if x1 <= x0 or y1 <= y0:
                manifest["crop_errors"].append(
                    f"{resource.resource_id}: reversed or empty box"
                )
                continue

            page = document[page_number - 1]
            page_rect = page.rect

            # Convert normalized coordinates to PDF points.
            rect = pymupdf.Rect(
                x0 / 1000 * page_rect.width,
                y0 / 1000 * page_rect.height,
                x1 / 1000 * page_rect.width,
                y1 / 1000 * page_rect.height,
            )

            # Add a small margin so labels are not cut off.
            rect = rect + (-12, -12, 12, 12)
            rect = rect & page_rect

            filename = (
                safe_filename(resource.resource_id) + ".png"
            )

            image_path = OUTPUT_DIR / filename

            pixmap = page.get_pixmap(
                matrix=pymupdf.Matrix(2, 2),
                clip=rect,
                alpha=False,
            )

            pixmap.save(image_path)

            item = resource.model_dump()
            item["image"] = str(
                image_path.relative_to(BASE_DIR)
            )

            manifest["resources"].append(item)

            print(
                f"Saved: {filename} "
                f"(page {page_number})"
            )

        document.close()

        with manifest_path.open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(
                manifest,
                file,
                indent=2,
                ensure_ascii=False,
            )

        print("\nResource extraction completed.")
        print(f"Resources identified: {len(extraction.resources)}")
        print(f"Crops saved: {len(manifest['resources'])}")
        print(f"Crop errors: {len(manifest['crop_errors'])}")
        print(f"Manifest: {manifest_path}")

    finally:
        if uploaded_file is not None:
            try:
                client.files.delete(name=uploaded_file.name)
                print("Temporary uploaded file deleted.")
            except Exception as error:
                print(f"Could not delete temporary file: {error}")

        client.close()


if __name__ == "__main__":
    main()
