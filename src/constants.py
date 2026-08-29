"""Centralized constants for the OMH RAG pipeline."""

DATA_DIR = "data"
DB_DIR = ".chroma_db"

POLICY_URLS = [
    "https://omh.ny.gov/omhweb/policymanual/i-100intranet.pdf",
    "https://omh.ny.gov/omhweb/policymanual/om-500.pdf",
    "https://omh.ny.gov/omhweb/policymanual/om-505.pdf",
    "https://omh.ny.gov/omhweb/policymanual/pc-522.pdf",
]

# Top margins are dialed in tightly to remove headers without clipping body text.
PDF_CROP_MARGINS = {
    "I-100INTRANET": {"top": 0.10, "bottom": 1.0},
    "OM-500": {"top": 0.10, "bottom": 1.0},
    "OM-505": {"top": 0.10, "bottom": 1.0},
    "PC-522": {"top": 0.09, "bottom": 1.0},
}
DEFAULT_CROP_MARGIN_TOP = 0.10
DEFAULT_CROP_MARGIN_BOTTOM = 1.0

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CHUNK_SEPARATORS = ["\n\n", "\n", " ", ""]

EMBEDDING_MODEL = "models/gemini-embedding-2"
LLM_MODEL = "gemini-3.1-flash-lite"
LLM_TEMPERATURE = 0.2
RETRIEVER_K = 5

EXIT_COMMANDS = {"quit", "exit", "q"}
