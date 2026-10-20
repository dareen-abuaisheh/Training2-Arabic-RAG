""" Reviewed repairs for exact versions of the two existing extractions."""

import re
IMAGE_LINE = re.compile(r"!\[(?P<caption>.*)\]\((?P<target>[^\n]+)\)")

# SHA-256 identifies the original file bytes, not just a reusable filename.
# Any changed or new extraction receives no document-specific repairs.
REVIEWED_SOURCE_HASHES = {
    "326872b6570437706ec3ef9b174d109fe9173748e39cee4d982f8b97f6bbf640",
    "e40f82afd7cd13145151238914a2309f950b7fa5c61a80f1e294393b790528cd",
}

# These image references and captions were checked against the current PDFs.
# Exact matches avoid guessing whether an English paragraph is source text.
REVIEWED_IMAGES = {
    "2dfa6ac3edfe874f68aa0cbccaa42322_img.jpg",
    "2173c0102e23a5ff29d90d4353fc0339_img.jpg",
    "1df14e8a75f3f3fcf750749dbbd2c3f1_img.jpg",
    "48f188337e3ba41df38fab9ac0afb1bd_img.jpg",
    "dc1e692e5b7efd8e5a2a790c458646d9_img.jpg",
    "acfc53eca625d62b38aa2563efa95c3e_img.jpg",
    "3267a096e9ca525744d8cd820f12eb59_img.jpg",
}

REVIEWED_CAPTIONS = (
    "A stylized illustration of a human brain, where the cerebral cortex is depicted "
    "as a complex network of circuit boards. The brain is flanked by various scenes "
    "of modern life: on the left, a person's face is shown with a facial recognition "
    "grid and binary code; below that, a person is looking at a smartphone. In the "
    "center, a person is sitting on a laptop, and another is shopping with a cart. "
    "On the right, a person is sitting on a large stack of coins, and another is "
    "working on a laptop. A security camera and a cell tower are also present. The "
    "entire scene is interconnected by orange lines and binary digits (0s and 1s), "
    "symbolizing the flow of data and the integration of technology into daily life.",
    "A two-panel graphic showing a glowing fingerprint with the text 'IDS vs IPS' "
    "on the left, and a woman's face overlaid with facial recognition grid lines "
    "and digital interface overlays on the right.",
    "Side-by-side comparison of a man in a suit, showing the original on the left "
    "and a deepfake version on the right with callout lines pointing to fewer "
    "wrinkles, darker iris, and fuller lips.",
    "A man in a shirt and tie carries a cardboard box with a potted plant and a "
    "folder over his shoulder in an office setting, with a white humanoid robot "
    "standing in the foreground on the left.",
    "A white humanoid robot stands on the top step of a metal staircase on a "
    "stage, with a man in a blue shirt standing to the right against a blue "
    "backdrop displaying technical schematics.",
)

# This generated diagram omitted relationships. Preserve its source labels below
# it, but do not let the incomplete graph become evidence for RAG answers.
INCOMPLETE_DIAGRAM = (
    "```\n"
    "graph LR; A[حل المعضلات] --- B[إستخدام المعرفة]; A --- C[تمثيل المعرفة]; "
    "D[المنطق و الإستدلال] --- B; D --- C; E[التخطيط] --- C; "
    "F[معالجة عدم التأكد] --- C; G[التعلم] --- H[الحصول على المعرفة];\n"
    "```"
)

TITLE_FIXES = {
    "# الذكاء الاصطناعيالمحاضرة الثانية": "# الذكاء الاصطناعي\n\n## المحاضرة الثانية",
}


def comparison_key(text: str) -> str:
    """Ignore whitespace differences only when matching a reviewed paragraph."""
    return " ".join(text.split())


def apply_reviewed_fixes(text: str, source_hash: str) -> tuple[str, dict[str, int]]:
    """Apply exact repairs only when the original extraction has been reviewed."""
    if source_hash not in REVIEWED_SOURCE_HASHES:
        return text, {}

    text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    captions = {comparison_key(caption) for caption in REVIEWED_CAPTIONS}
    counts = {"images": 0, "captions": 0, "diagrams": 0, "titles": 0}
    lines = []

    for line in text.splitlines():
        match = IMAGE_LINE.fullmatch(line.strip())
        if match and match["target"] in REVIEWED_IMAGES:
            captions.add(comparison_key(match["caption"]))
            counts["images"] += 1
        else:
            lines.append(line)

    # Split at blank lines, preserving lists, tables, and Markdown hard breaks.
    blocks = re.split(r"\n[ \t]*\n", "\n".join(lines))
    kept_blocks = []
    for block in blocks:
        block = block.strip("\n")
        if not block.strip():
            continue
        if comparison_key(block) in captions:
            counts["captions"] += 1
        elif comparison_key(block) == comparison_key(INCOMPLETE_DIAGRAM):
            counts["diagrams"] += 1
        elif block in TITLE_FIXES:
            kept_blocks.append(TITLE_FIXES[block])
            counts["titles"] += 1
        else:
            kept_blocks.append(block)

    cleaned = "\n\n".join(kept_blocks).rstrip() + "\n"
    if not cleaned.strip():
        raise ValueError("Cleaning produced empty text; output was not saved.")
    return cleaned, counts
