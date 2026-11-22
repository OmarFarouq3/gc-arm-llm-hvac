"""Knowledge base and retrieval.

Paper: Section 3.4. Sources: design/nameplate data, normal operating ranges,
maintenance records and expert rules.

How it works
1. load_documents() reads knowledge_base/*.md and cuts each file into small chunks
   (one bullet point or one short paragraph each), remembering the source file.
2. build_index() turns every chunk into a vector with a sentence-transformers model
   and puts the vectors into a FAISS index (cosine similarity). The index is cached
   in .cache/rag/ and only rebuilt when the documents or the model change.
3. retrieve() embeds the rule text and returns the k most similar chunks.

The glossary (knowledge_base/glossary.md) is NOT indexed. Rule items use codes like
"CT_SW_TEMP_1=high" that an embedding model does not understand, so describe_variables()
looks the codes up in the glossary instead; its output is used (a) to expand the
retrieval query and (b) as variable definitions in the LLM context.

Owner: Member C
"""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
KB_DIR = ROOT / "knowledge_base"
CACHE_DIR = ROOT / ".cache" / "rag"          # git-ignored
GLOSSARY_FILE = "glossary.md"

_CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
DEFAULT_MODEL = _CONFIG["rag"]["embedding_model"]
DEFAULT_TOP_K = _CONFIG["rag"]["top_k"]

# Loaded once per Python process (see _get_index / _get_model).
_models = {}
_state = {"index": None, "chunks": None}


# ---------------------------------------------------------------------------
# 1. Documents -> chunks
# ---------------------------------------------------------------------------

def load_documents(folder=KB_DIR, exclude=(GLOSSARY_FILE,)):
    """Split every .md file in `folder` into chunks.

    A chunk is one bullet point ("- ...", continuation lines included) or one
    paragraph. Headings are not chunks; the nearest heading is kept as "section".

    Returns a list of dicts: {"id", "source", "section", "text"}.
    """
    chunks = []
    for path in sorted(Path(folder).glob("*.md")):
        if path.name in exclude:
            continue
        section = ""
        current = []

        def flush():
            text = " ".join(current).strip()
            if text:
                chunks.append({"id": len(chunks), "source": path.name,
                               "section": section, "text": text})
            current.clear()

        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):              # heading: new section
                flush()
                section = stripped.lstrip("#").strip()
            elif not stripped:                        # blank line ends a paragraph
                flush()
            elif re.match(r"^[-*]\s+", stripped):     # new bullet
                flush()
                current.append(re.sub(r"^[-*]\s+", "", stripped))
            else:                                     # continuation / paragraph text
                current.append(stripped)
        flush()
    return chunks


# ---------------------------------------------------------------------------
# 2. Chunks -> FAISS index (cached)
# ---------------------------------------------------------------------------

def _get_model(model_name):
    """Load a sentence-transformers model once and reuse it."""
    if model_name not in _models:
        from sentence_transformers import SentenceTransformer
        _models[model_name] = SentenceTransformer(model_name)
    return _models[model_name]


def _embed(texts, model_name):
    """Unit-length float32 vectors, so inner product = cosine similarity."""
    vecs = _get_model(model_name).encode(list(texts), normalize_embeddings=True,
                                         show_progress_bar=False)
    return np.asarray(vecs, dtype="float32")


def build_index(chunks, model_name=DEFAULT_MODEL, cache_dir=CACHE_DIR):
    """Embed the chunks and build a FAISS inner-product index.

    The cache key is a hash of the model name and all chunk texts, so editing any
    knowledge-base file (or switching model) triggers a rebuild automatically.
    Returns the FAISS index; row i of the index is chunks[i].
    """
    import faiss

    key = hashlib.sha256(json.dumps([model_name, [c["source"] + c["text"] for c in chunks]])
                         .encode("utf-8")).hexdigest()[:16]
    cache_dir = Path(cache_dir)
    index_path = cache_dir / f"index_{key}.faiss"

    if index_path.exists():
        index = faiss.read_index(str(index_path))
    else:
        vecs = _embed([c["text"] for c in chunks], model_name)
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        cache_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_path))

    _state.update(index=index, chunks=chunks, model=model_name)
    return index


def _get_index():
    """Return (index, chunks), building / loading the default index on first use."""
    if _state["index"] is None:
        build_index(load_documents(), DEFAULT_MODEL)
    return _state["index"], _state["chunks"]


# ---------------------------------------------------------------------------
# 3. Glossary lookup for variable codes
# ---------------------------------------------------------------------------

def _load_glossary(folder=KB_DIR):
    """{"CT_SW_TEMP": "water leaving ... (degF)", ...} from glossary.md."""
    path = Path(folder) / GLOSSARY_FILE
    glossary = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^\s*-\s*([A-Z0-9_]+):\s*(.+)$", line)
            if m:
                glossary[m.group(1)] = m.group(2).strip()
    return glossary


def describe_variables(text):
    """Find variable codes in `text` and return one definition line per code,
    e.g. "CT_SW_TEMP_1 = water leaving (supply) temperature of a cooling tower (degF), unit 1".
    Unknown names (e.g. the paper's "Cooling capacity") are skipped."""
    glossary = _load_glossary()
    lines, seen = [], set()
    for code in re.findall(r"\b([A-Z][A-Z0-9_]*[A-Z0-9])\b", text):
        if code in seen:
            continue
        seen.add(code)
        base, unit = code, None
        m = re.match(r"^(.*)_(\d)$", code)
        if code not in glossary and m:             # strip unit number _1.._3
            base, unit = m.group(1), m.group(2)
        if base in glossary:
            lines.append(f"{code} = {glossary[base]}" + (f", unit {unit}" if unit else ""))
    return lines


# ---------------------------------------------------------------------------
# 4. Retrieval
# ---------------------------------------------------------------------------

def retrieve(rule_text, k=DEFAULT_TOP_K):
    """Return the k knowledge chunks most similar to the rule.

    The query is the rule text plus the plain-English meaning of its variable
    codes (from the glossary), which makes retrieval work for coded names.
    Each result is a chunk dict with an extra "score" (cosine similarity).
    """
    index, chunks = _get_index()
    query = rule_text
    definitions = describe_variables(rule_text)
    if definitions:
        query += "\n" + "\n".join(definitions)
    scores, ids = index.search(_embed([query], _state["model"]), min(k, len(chunks)))
    return [{**chunks[i], "score": float(s)} for s, i in zip(scores[0], ids[0]) if i >= 0]


if __name__ == "__main__":
    for c in retrieve("Antecedent: CT_FAN_SPD_1=high; Consequent: CT_SW_TEMP_1=high"):
        print(f"{c['score']:.3f}  [{c['source']}]  {c['text'][:100]}")
