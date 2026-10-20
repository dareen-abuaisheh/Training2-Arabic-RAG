import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "original"
OUTPUT_DIR = ROOT / "data" / "extracted"
API_URL = "https://www.datalab.to/api/v1/convert"


def extract_pdf(pdf_path: Path, session: requests.Session, headers: dict) -> None:
    """Submit a PDF, wait for Chandra, and save only the extracted text."""
    output_path = OUTPUT_DIR / pdf_path.with_suffix(".txt").name
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Already extracted: {pdf_path.name}", flush=True)
        return

    print(f"Uploading: {pdf_path.name}", flush=True)
    with pdf_path.open("rb") as pdf_file:
        response = session.post(
            API_URL,
            headers=headers,
            files={"file": (pdf_path.name, pdf_file, "application/pdf")},
            data={
                "mode": "accurate",
                "output_format": "markdown",
                "paginate": "true",
                "disable_image_captions": "true",
            },
            timeout=(10, 120),
        )

    response.raise_for_status()
    job = response.json()
    if job.get("success") is False:
        raise RuntimeError(job.get("error") or "Submission failed.")
    check_url = job.get("request_check_url")
    if not check_url:
        raise RuntimeError("Datalab did not return a job tracking URL.")

    # Send the API key only to the expected Datalab polling host.
    from urllib.parse import urlsplit

    location = urlsplit(check_url)
    if location.scheme != "https" or location.netloc not in {"www.datalab.to", "api.datalab.to"}:
        raise RuntimeError("Datalab returned an unexpected job tracking address.")

    print(f"Reading pages with Chandra: {pdf_path.name}", flush=True)
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        response = session.get(check_url, headers=headers, timeout=(10, 60))
        response.raise_for_status()
        result = response.json()
        if result.get("success") is False or result.get("status") == "failed":
            raise RuntimeError(result.get("error") or "Processing failed.")

        if result.get("status") == "complete":
            if result.get("result_url"):
                # Signed result downloads do not need the API key.
                download = session.get(result["result_url"], timeout=(10, 60))
                download.raise_for_status()
                result = {
                    **download.json(),
                    **{key: value for key, value in result.items() if value is not None},
                }
            if result.get("success") is False:
                raise RuntimeError(result.get("error") or "Processing failed.")
            if (result.get("metadata") or {}).get("failed_pages"):
                raise RuntimeError("Some pages failed. Output was not saved.")

            text = result.get("markdown")
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError("Datalab returned no extracted text.")
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            output_path.write_text(text, encoding="utf-8")
            print(f"Saved: data/extracted/{output_path.name}", flush=True)
            return
        time.sleep(2)
    raise TimeoutError("Processing exceeded the 15-minute waiting limit.")


def main() -> None:
    load_dotenv(ROOT / ".env", encoding="utf-8-sig")
    api_key = os.getenv("DATALAB_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("DATALAB_API_KEY is missing from .env.")

    pdf_files = sorted(SOURCE_DIR.glob("*.pdf"))
    if not pdf_files:
        raise SystemExit("No PDFs found in data/original/.")

    try:
        with requests.Session() as session:
            for pdf_path in pdf_files:
                extract_pdf(pdf_path, session, {"X-API-Key": api_key})
    except requests.Timeout:
        raise SystemExit(
            "The network request timed out. Check the Datalab dashboard before retrying; "
            "the job may already have been accepted."
        ) from None
    except requests.HTTPError as error:
        status = error.response.status_code if error.response is not None else "unknown"
        raise SystemExit(f"Datalab returned HTTP {status}. No new output was saved for that job.") from None
    except requests.RequestException:
        raise SystemExit("Could not reach Datalab. Check your connection before retrying.") from None
    except (RuntimeError, TimeoutError, ValueError) as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    main()
