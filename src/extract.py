
from pathlib import Path
import pymupdf
import json

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "PDFS"
OUTPUT_DIR = BASE_DIR / "output" / "extracted"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

report = []

for pdf_path in sorted(PDF_DIR.rglob("*.pdf")):

    relative_path = pdf_path.relative_to(PDF_DIR)
    output_folder = OUTPUT_DIR / relative_path.parent
    output_folder.mkdir(parents=True, exist_ok=True)

    output_path = output_folder / f"{pdf_path.stem}.json"

    try:
        with pymupdf.open(pdf_path) as pdf:

            pages = []

            for page_number, page in enumerate(pdf, start=1):

                page_data = page.get_text("dict", sort=True)
                blocks = []

                for block in page_data["blocks"]:

                    # Text blocks
                    if block["type"] == 0:

                        text = "".join(
                            span["text"]
                            for line in block["lines"]
                            for span in line["spans"]
                        ).strip()

                        if text:
                            blocks.append({
                                "type": "text",
                                "bbox": block["bbox"],
                                "text": text
                            })

                    # Image blocks
                    elif block["type"] == 1:
                        blocks.append({
                            "type": "image",
                            "bbox": block["bbox"],
                            "width": block.get("width"),
                            "height": block.get("height")
                        })

                pages.append({
                    "page_number": page_number,
                    "width": page.rect.width,
                    "height": page.rect.height,
                    "text": page.get_text("text", sort=True),
                    "blocks": blocks
                })

            document = {
                "source_file": pdf_path.name,
                "source_path": str(relative_path),
                "total_pages": len(pdf),
                "pages": pages
            }

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(document, file, ensure_ascii=False, indent=2)

        report.append({
            "file": str(relative_path),
            "pages": len(pages),
            "status": "SUCCESS"
        })

        print(f"Processed: {relative_path}")

    except Exception as error:
        report.append({
            "file": str(relative_path),
            "status": "ERROR",
            "error": str(error)
        })
        print(f"ERROR: {relative_path}: {error}")

with (OUTPUT_DIR / "extraction_report.json").open(
    "w", encoding="utf-8"
) as file:
    json.dump(report, file, indent=2)

print(f"\nProcessed: {len(report)} PDFs")

