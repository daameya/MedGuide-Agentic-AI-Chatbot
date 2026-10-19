import argparse
import hashlib
from pathlib import Path
import fitz
from app.knowledge import ensure_index, embed


def chunks(text, size=1100, overlap=180):
    text = " ".join(text.split())
    step = size - overlap

    return [
        text[i:i + size]
        for i in range(0, len(text), step)
        if len(text[i:i + size]) > 80
    ]


def upload_missing(index, batch):
    if not batch:
        return

    # Check which chunks already exist in Pinecone
    existing = index.fetch(ids=[item[0] for item in batch])
    existing_ids = set(existing.vectors.keys())

    # Only process chunks that have not been uploaded
    pending = [item for item in batch if item[0] not in existing_ids]
    skipped = len(batch) - len(pending)

    if not pending:
        print(f"Skipped {skipped} existing chunks.", flush=True)
        return

    # Generate embeddings for missing chunks
    vectors = embed([item[1] for item in pending])

    # Upload embeddings and metadata to Pinecone
    index.upsert(
        vectors=[
            {
                "id": item[0],
                "values": vector,
                "metadata": item[2],
            }
            for item, vector in zip(pending, vectors)
        ]
    )

    print(
        f"Uploaded {len(pending)} new chunks; "
        f"skipped {skipped} existing chunks.",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default="data/medical_book.pdf")
    args = parser.parse_args()

    pdf = Path(args.pdf)

    if not pdf.exists():
        raise SystemExit(f"PDF not found: {pdf}")

    index = ensure_index()

    with fitz.open(pdf) as doc:
        batch = []

        for page_number, page in enumerate(doc, start=1):
            for part, content in enumerate(
                chunks(page.get_text()), start=1
            ):
                key = hashlib.sha256(
                    f"{pdf.name}:{page_number}:{part}:{content}".encode()
                ).hexdigest()

                metadata = {
                    "page": page_number,
                    "text": content,
                    "file": pdf.name,
                }

                batch.append((key, content, metadata))

                if len(batch) >= 50:
                    upload_missing(index, batch)
                    batch = []

            if page_number % 50 == 0:
                print(
                    f"Processed pages: {page_number}/{len(doc)}",
                    flush=True,
                )

        # Process the final partial batch
        if batch:
            upload_missing(index, batch)

    print("Ingestion complete.", flush=True)


if __name__ == "__main__":
    main()