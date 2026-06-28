from __future__ import annotations

from pathlib import Path

# Pipeline lives at RegConflict/src/pipeline/, so go up 3 levels: pipeline → src → RegConflict
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Corpus is organised by license tier in RegConflict (not by jurisdiction directly as in TRIAG)
CORPUS_ROOT = PROJECT_ROOT / "data" / "corpus"
DATA_ROOT = PROJECT_ROOT / "data"
EXTRACTED_DIR = DATA_ROOT / "extracted"
QC_DIR = DATA_ROOT / "qc_reports"
CHUNKS_DIR = DATA_ROOT / "chunks"

# Chunking targets (tokens, cl100k via tiktoken)
CHUNK_TARGET_TOKENS = 200
CHUNK_MAX_TOKENS = 400

# QC: sample 5-10 random pages per doc per pipeline spec
QC_MIN_SAMPLE = 5
QC_MAX_SAMPLE = 10

# Source registry: maps (jurisdiction, issuing_body) -> parser id + citation unit label
# Parser id is matched to a parser in pipeline.parse. Unknown -> "generic".
SOURCE_REGISTRY = {
    # Singapore
    ("Singapore", "MAS-Notices"):       ("mas_notices",  "clause"),
    ("Singapore", "MAS-Guidelines"):    ("mas_notices",  "paragraph"),
    ("Singapore", "MAS-Consultations"): ("generic",      "section"),
    ("Singapore", "IRAS"):              ("generic",      "paragraph"),
    ("Singapore", "SSO"):               ("au_act",       "section"),  # SSO Acts mirror s.N(N) style
    # Australia
    ("Australia", "ASIC"):              ("asic_rg",      "paragraph"),
    ("Australia", "APRA"):              ("generic",      "paragraph"),
    ("Australia", "AUSTRAC"):           ("generic",      "section"),
    ("Australia", "ATO"):               ("generic",      "paragraph"),
    ("Australia", "OAIC"):              ("generic",      "paragraph"),
    ("Australia", "RBA"):               ("generic",      "section"),
    ("Australia", "Treasury"):          ("generic",      "section"),
    ("Australia", "FederalRegister"):   ("au_act",       "section"),
    # EU
    ("EU", "EUR-Lex"):                  ("eu_regulation","article"),
    ("EU", "EBA-MiCA"):                 ("eu_regulation","article"),
    ("EU", "ESMA-Guidelines"):          ("generic",      "paragraph"),
    ("EU", "ESMA-Consultations"):       ("generic",      "section"),
    ("EU", "ESAs-Joint"):               ("generic",      "section"),
    # International
    ("International", "FATF"):          ("generic",      "recommendation"),
    ("International", "BASEL"):         ("generic",      "paragraph"),
    ("International", "IOSCO"):         ("generic",      "section"),
}

# Controlled vocabularies (copied verbatim from ingestion_pipeline.md)
REGIME_TAGS = [
    "AML_CTF",
    "MARKET_CONDUCT",
    "LICENSING",
    "PRUDENTIAL",
    "DATA_PROTECTION",
    "TAX",
    "SANCTIONS",
    "CONSUMER_PROTECTION",
]

TOPIC_TAGS = [
    "TRAVEL_RULE",
    "CUSTOMER_DUE_DILIGENCE",
    "BENEFICIAL_OWNERSHIP",
    "SUSPICIOUS_TRANSACTION_REPORTING",
    "REVERSE_SOLICITATION",
    "STABLECOIN_ISSUANCE",
    "CRYPTO_CUSTODY",
    "DESIGNATED_SERVICE_PROVISION",
    "CROSS_BORDER_TRANSFER",
    "RECORD_KEEPING",
    "ENFORCEMENT_ACTION_REFERENCE",
]


def source_of(path: Path) -> tuple[str, str]:
    """Return (jurisdiction, issuing_body) inferred from path under CORPUS_ROOT."""
    rel = path.resolve().relative_to(CORPUS_ROOT)
    parts = rel.parts
    return parts[0], parts[1]


def parser_id_for(jurisdiction: str, issuing_body: str) -> tuple[str, str]:
    return SOURCE_REGISTRY.get((jurisdiction, issuing_body), ("generic", "section"))
