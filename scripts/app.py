"""A simple local interface for Arabic questions, top-three chunks, and citations."""

import os
from pathlib import Path
from threading import Lock

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]


@st.cache_resource(show_spinner=False, max_entries=1)
def load_retriever(file_versions: tuple):
    """Reuse E5 and FAISS, reloading when the underlying artifacts change."""
    import torch
    from search import Retriever

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    # A lock protects the shared tokenizer when several browser tabs submit.
    return Retriever(), Lock()


def retrieve(question: str) -> list[dict]:
    paths = [ROOT / "data" / name for name in ("chunks.jsonl", "embeddings.npz", "index.faiss")]
    versions = tuple((path.stat().st_mtime_ns, path.stat().st_size) for path in paths)
    retriever, lock = load_retriever(versions)
    with lock:
        # Use only the three displayed passages as evidence for this interface.
        return retriever.search(question, top_k=3, include_neighbors=False)


def render_answer(answer: str, passages: list[dict]) -> None:
    """Make each citation clickable and name the file, page, and chunk it maps to."""
    from ask import CITATION

    linked_answer = CITATION.sub(lambda match: f"[المصدر {match[1]}](#source-{match[1]})", answer)
    with st.container(key="answer_text"):
        st.markdown(linked_answer)
    cited = sorted({int(value) for value in CITATION.findall(answer)})
    if cited:
        st.markdown("**المصادر المستخدمة في الإجابة**")
        for number in cited:
            passage = passages[number - 1]
            st.markdown(f"**المصدر {number}** · `{passage['source']}` · الصفحة {passage['page']}")
            st.caption(f"معرّف المقطع: `{passage['id']}`")


def render_passages(passages: list[dict]) -> None:
    st.subheader("أكثر ٣ مقاطع صلة بسؤالك")
    st.caption("مرتبة حسب التشابه مع سؤالك. درجة التشابه لا تعني مدى صحة الإجابة.")
    for rank, passage in enumerate(passages, start=1):
        # The anchor contains a generated integer, never raw document HTML.
        st.markdown(f'<span id="source-{rank}"></span>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(f"**المصدر {rank}** · `{passage['source']}`")
            st.caption(f"الصفحة {passage['page']} · درجة التشابه {passage['score']:.3f}")
            st.caption(f"معرّف المقطع: `{passage['id']}`")
            with st.expander("قراءة المقطع", expanded=rank == 1):
                with st.container(key=f"chunk_text_{rank}"):
                    # Markdown renders OCR headings and lists; source HTML stays disabled.
                    st.markdown(passage["text"])


def main() -> None:
    st.set_page_config(page_title="اسأل مستنداتك", page_icon="📚", layout="centered")
    st.markdown("""
        <style>
        :root { color-scheme: light; }
        .stApp, [data-testid="stHeader"], [data-testid="stMain"] {
            background: #fbf8f3 !important; color: #354b48;
            font-family: "Segoe UI", Tahoma, Arial, sans-serif;
        }
        .block-container {
            max-width: 920px; padding-top: 2.5rem;
            direction: rtl; text-align: right;
        }
        .stApp h1, .stApp h2, .stApp h3, .stApp p,
        .stApp label, .stApp li, .stApp summary {
            color: #354b48; text-align: right;
        }
        .stApp [data-testid="stCaptionContainer"] p { color: #687873; }
        [data-testid="stForm"], [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: #e1d9cd;
        }
        .stApp [data-testid="stForm"] {
            background: #fffdf9; border: 1px solid #e1d9cd; border-radius: 18px;
        }
        .stApp [data-testid="stTextAreaRootElement"],
        .stApp [data-testid="stTextArea"] [data-baseweb="textarea"],
        .stApp [data-testid="stTextArea"] [data-baseweb="base-input"],
        .stApp textarea {
            background: #f3ece2 !important; color: #354b48 !important;
            border-color: #ded4c7 !important;
            direction: rtl; text-align: right;
            caret-color: #386f69;
        }
        .stApp textarea::placeholder { color: #747c74 !important; opacity: 1; }
        .stApp [data-testid="stTextAreaRootElement"]:focus-within {
            outline: 2px solid #a8c7c1; outline-offset: 2px;
        }
        .stApp [data-testid="stFormSubmitButton"] button,
        .stApp [data-testid="stBaseButton-primary"] {
            background: #bbd7d2 !important; border: 1px solid #a8c7c1 !important;
            color: #254f4c !important; border-radius: 12px; box-shadow: none !important;
        }
        .stApp [data-testid="stFormSubmitButton"] button p,
        .stApp [data-testid="stBaseButton-primary"] p { color: #254f4c !important; }
        .stApp [data-testid="stFormSubmitButton"] button:hover,
        .stApp [data-testid="stBaseButton-primary"]:hover {
            background: #a9cbc4 !important; border-color: #91b9b1 !important;
        }
        .stApp [data-testid="stFormSubmitButton"] button:focus-visible {
            outline: 2px solid #7da9a1; outline-offset: 3px;
        }
        .stApp [data-testid="stExpander"] details {
            background: #fffdf9 !important; border-color: #e1d9cd !important;
        }
        .stApp [data-testid="stAlertContainer"] {
            background: #e7f1ed !important; color: #354b48 !important;
            border: 1px solid #d4e3dd; border-radius: 14px;
        }
        .stApp [data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) {
            background: #f6efdf !important; border-color: #e8dcc0;
        }
        .stApp [data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) {
            background: #f5e9e3 !important; border-color: #e7d1c6;
        }
        .stApp a { color: #386f69; }
        .stApp code {
            direction: ltr; unicode-bidi: isolate;
            background: #eee8de; color: #495b54;
        }
        .st-key-answer_text, [class*="st-key-chunk_text_"] {
            direction: rtl; text-align: right; line-height: 1.9;
            overflow-wrap: anywhere;
        }
        .st-key-answer_text p, [class*="st-key-chunk_text_"] p {
            unicode-bidi: plaintext;
        }
        </style>
        """, unsafe_allow_html=True)
    st.title("اسأل مستنداتك")
    st.caption("اطرح سؤالك بالعربية، واقرأ الإجابة مع المقاطع والمصادر التي تستند إليها.")

    with st.form("question_form"):
        question = st.text_area("سؤالك", placeholder="اكتب سؤالك هنا...", height=100)
        submitted = st.form_submit_button("اعرض الإجابة", type="primary", width="stretch")

    if submitted:
        # Clear previous results so a failed new question cannot display an old answer.
        st.session_state.pop("result", None)
        if not question.strip():
            st.warning("يرجى كتابة سؤال أولاً.")
        else:
            result = {"question": question.strip(), "passages": [], "answer": None, "error": None}
            st.session_state["result"] = result
            try:
                with st.spinner("جارٍ البحث عن أكثر ثلاثة مقاطع صلة بسؤالك...", show_time=False):
                    result["passages"] = retrieve(result["question"])
            except Exception as error:
                result["error"] = str(error)

    result = st.session_state.get("result")
    if result is None:
        st.info("ستظهر هنا إجابتك مع أكثر ثلاثة مقاطع صلة بسؤالك.")
        return

    st.divider()
    st.caption(f"نتائج سؤالك: {result['question']}")
    st.subheader("الإجابة")
    answer_area = st.container()
    if result["passages"]:
        st.divider()
        render_passages(result["passages"])

    if submitted and result["passages"] and result["error"] is None:
        try:
            from ask import generate_answer

            with answer_area:
                with st.spinner("جارٍ إعداد إجابتك. قد يستغرق ذلك بضع دقائق...", show_time=False):
                    result["answer"] = generate_answer(result["question"], result["passages"])
        except Exception as error:
            result["error"] = str(error)

    with answer_area:
        if result["error"]:
            st.error("تعذر إكمال الطلب. تحقق من ملفات المشروع وتشغيل Ollama، ثم حاول مرة أخرى.")
            with st.expander("تفاصيل الخطأ"):
                st.code(result["error"], language="text")
        elif result["answer"]:
            render_answer(result["answer"], result["passages"])


if __name__ == "__main__":
    main()
