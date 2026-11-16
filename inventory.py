from pathlib import Path
import re
import csv
import fitz  # PyMuPDF

BASE_DIR = Path(__file__).parent
PDF_DIR = BASE_DIR / "PDFS"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(exist_ok=True)

# Recognise Cambridge filenames
pattern = re.compile(
    r"(?P<code>\d{4})_"
    r"(?P<session>[a-z]\d{2})_"
    r"(?P<type>qp|ms)_"
    r"(?P<component>\d{2})\.pdf$",
    re.IGNORECASE
)

documents = []

for path in sorted(PDF_DIR.rglob("*.pdf")):
    match = pattern.match(path.name)

    if not match:
        print(f"UNRECOGNISED: {path}")
        continue

    info = match.groupdict()

    try:
        with fitz.open(path) as pdf:
            page_count = len(pdf)
            metadata = pdf.metadata
            text_chars = sum(
                len(page.get_text().strip())
                for page in pdf
            )

        documents.append({
            "filename": path.name,
            "path": str(path.relative_to(BASE_DIR)),
            "subject_folder": path.parent.name,
            "subject_code": info["code"],
            "session": info["session"],
            "document_type": info["type"].upper(),
            "component": info["component"],
            "pages": page_count,
            "extracted_text_characters": text_chars,
            "text_status": (
                "TEXT_AVAILABLE" if text_chars > 0
                else "POSSIBLE_SCANNED_PDF"
            ),
            "pdf_title": metadata.get("title") or ""
        })

    except Exception as error:
        print(f"ERROR reading {path}: {error}")

# Export CSV inventory
if documents:
    with open(
        OUTPUT_DIR / "document_inventory.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=documents[0].keys()
        )
        writer.writeheader()
        writer.writerows(documents)

# Summary
print(f"\nTotal PDFs: {len(documents)}")

for subject in sorted(set(d["subject_folder"] for d in documents)):
    items = [d for d in documents if d["subject_folder"] == subject]

    print(
        f"{subject}: "
        f"{sum(d['document_type'] == 'QP' for d in items)} QPs, "
        f"{sum(d['document_type'] == 'MS' for d in items)} MS"
    )

print(f"\nInventory saved in: {OUTPUT_DIR / 'document_inventory.csv'}")
