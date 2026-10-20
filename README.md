# Arabic RAG System

**Dareen Abuaisheh · Task 2**

A learning project that answers Arabic questions using evidence from real PDFs. This README is the short implementation overview accompanying my **Task 2 – Arabic RAG System** documentation.

## Documents and technologies

- **File1-Scanned.pdf — 4 scanned pages:** AI's effects on society, security, international relations, employment, and the future.
- **File2-Mixed.pdf — 19 mixed-content pages:** AI branches, programming, reasoning, knowledge representation, planning, machine learning, NLP, and robotics.

The 23 PDF pages produced **22 searchable text chunks**. The final image-only page is excluded.

| Stage | Technologies |
| --- | --- |
| Text extraction | Hosted Chandra through Datalab, Requests, python-dotenv |
| Cleaning | Python, regular expressions, Unicode normalization |
| Embeddings | `intfloat/multilingual-e5-small`, Hugging Face Transformers, PyTorch CPU, NumPy |
| Retrieval | FAISS CPU with `IndexFlatIP` and normalized vectors for cosine similarity |
| Answer generation | Ollama with `qwen3:4b-instruct-2507-q4_K_M` |
| Interface | Streamlit |

## How it works and key decisions

PDFs → Chandra extraction → cleaned text → page records → chunks → embeddings → FAISS → local Qwen answer

- Chandra handles both PDF types. Embeddings and answer generation run locally.
- Raw and cleaned text stay separate. Conservative cleaning and reviewed repairs preserve source meaning.
- E5's tokenizer measures chunks within a 384-token budget. Questions use `query:`; chunks use `passage:`.
- Citations identify the PDF, physical page, and chunk. Insufficient evidence should produce an explicit refusal.
- The UI uses exactly three chunks; the CLI can add neighbors. Qwen's instruction variant replaced the slow thinking-only variant for answering.

Code lives in `scripts/`; documents and outputs live in `data/`. The repository includes the original PDFs, extracted and cleaned text, prepared chunks, embeddings, FAISS index, and evaluation questions. Virtual environments, model caches, and API keys in `.env` are excluded.

## Run the interface

The shared project uses the existing document chunks, embeddings, and FAISS index. Users do **not** need to run extraction, cleaning, chunking, embedding, or indexing.

**One-time setup per laptop:** install Python 3.12 and Ollama, then execute from the project root in Bash. Create a new virtual environment:

```bash
python -m venv .venv
```

Activate it with `source .venv/Scripts/activate` on Windows (Git Bash), or `source .venv/bin/activate` on macOS/Linux. Then run:

```bash
python -m pip install -r requirements.txt
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

**Each time:** start Ollama and launch the UI with:

```bash
bash run.sh
```

Open **http://127.0.0.1:8501** for questions, answers, and the top three source chunks. No environment activation is needed for this launcher. E5 downloads once on the first question unless its cache is already present; the app automatically embeds each new question to search the saved document vectors.

The prepared `data/chunks.jsonl`, `data/embeddings.npz`, and `data/index.faiss` files are included, so evaluators can launch the UI after the one-time setup. Original PDFs, extracted/cleaned text, and test questions are also included for evaluation; a Datalab key is only needed to rerun extraction.

All file paths resolve from the project folder, regardless of its location or the terminal's working directory. PDFs go in `data/original/`; extraction writes to `data/extracted/`, and cleaning writes to `data/cleaned/`. Extraction reads `DATALAB_API_KEY` from the project's `.env` file.

Only the project maintainer needs to rebuild these files after changing the documents. With the environment activated, run:

```bash
python scripts/extract.py
python scripts/clean.py --reviewed-fixes
python scripts/chunk.py
python scripts/embed.py
python scripts/build_index.py
```

Each laptop needs its own Python environment and Ollama model; E5 uses the project's `.cache/`. Do not share `.venv/` or your `.env` API key.

## Evaluation results

`data/test_questions.json` holds **30 text-only questions**, reference answer points, and source pages. Reference answers are excluded from retrieval.

**Evaluation date: 2026-10-20.** The latest five-question run was combined with 25 additional runs using the same UI settings: top three chunks, no neighboring context, and the same local Qwen instruction model. The questions, references, and pipeline were held fixed during the additional runs.

Answers were reviewed for semantic coverage, grounding, and citations; equivalent wording and concise answers were accepted. **Passed** means the requested answer was correct and supported. **Partial** means useful content with an omission, unsupported detail, citation issue, or contradiction. **Failed** means a wrong key answer or refusal despite sufficient evidence.

**Results: 17 Passed · 10 Partial · 3 Failed.** All four unanswerable cases correctly refused to invent missing information. Retrieval included all reference pages for **23 of 26 answerable questions**; finding the right pages did not always produce a complete answer.

| ID | Question focus | Result | Review note |
| --- | --- | --- | --- |
| q001 | Social media personalization | Partial | Core points correct; added filter-bubble details beyond retrieved text. |
| q002 | Isolation and mental health | Passed | — |
| q003 | Biometric identity checks | Passed | — |
| q004 | Diplomacy versus propaganda | Passed | — |
| q005 | Amazon and Google Health | Passed | — |
| q006 | Superintelligence and ethics | Passed | — |
| q007 | AI branches | Passed | — |
| q008 | Programming languages | Passed | — |
| q009 | Algorithms and mental reasoning | Passed | — |
| q010 | Intuition and symbolic processing | Partial | Knowledge-representation role omitted; added relationship not explicit in sources. |
| q011 | Problem-solving methods | Passed | — |
| q012 | Birds and rule exceptions | Partial | Exceptions explained; qualification problem and formal-representation difficulty omitted. |
| q013 | Ontology engineering and reading | Partial | Core answer correct; page-11 examples attributed only to page 13. |
| q014 | Implicit knowledge | Partial | Implicit knowledge/examples correct; its support for conscious symbolic knowledge omitted. |
| q015 | Machine-learning types | Partial | Correct full comparison followed by a contradictory insufficient-evidence refusal. |
| q016 | NLP capabilities and applications | Passed | — |
| q017 | Robot navigation concepts | Passed | — |
| q018 | Three security applications | Failed | Required page 3 was missing; cyber espionage replaced video surveillance. |
| q019 | Automation and employment | Failed | Refused despite both required pages being retrieved. |
| q020 | Uncertainty and computational cost | Partial | Missing page 5; uncertainty approach replaced by default-reasoning discussion. |
| q021 | Knowledge, ontology, and exceptions | Partial | Missing page 12; exception explanation unsupported by cited page 11. |
| q022 | Search and adaptive planning | Failed | Names uncertainty reasoning instead of planning as the second requested method. |
| q023 | NLP and social interaction | Partial | Correct applications, social effects, and both sources; reading/understanding human language omitted. |
| q024 | Robotics and Amazon employment | Partial | Robot navigation and Amazon impact correct; sensor-based obstacle avoidance omitted. |
| q025 | Planning with multiple agents | Passed | — |
| q026 | Computational learning theory | Passed | — |
| q027 | Missing algorithm and dataset names | Passed | — |
| q028 | Missing Amazon job-loss percentage | Passed | — |
| q029 | Missing Asimo date and speed | Passed | — |
| q030 | Missing RAG model recommendation | Passed | — |

The three failures illustrate different issues: q018 missed the video-surveillance page, q019 refused despite retrieving both required pages, and q022 selected the wrong method from available evidence. q015 also needs attention: its correct explanation was followed by a contradictory refusal.


