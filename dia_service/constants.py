"""Shared Dia2 service constants."""

DEFAULT_DIA_MODEL_SIZE = "1b"
DIA_MODEL_ID = "nari-labs/Dia2-1B"
DIA_MODEL_NAME = "Dia2-1B"
DIA_MODELS = {
    "1b": {
        "id": "nari-labs/Dia2-1B",
        "name": "Dia2-1B",
        "local_dir": "/models/dia2/local/Dia2-1B",
    },
    "2b": {
        "id": "nari-labs/Dia2-2B",
        "name": "Dia2-2B",
        "local_dir": "/models/dia2/local/Dia2-2B",
    },
}
MIMI_MODEL_ID = "kyutai/mimi"
DIA_LOCAL_DIR = "/models/dia2/local/Dia2-1B"
MIMI_LOCAL_DIR = "/models/dia2/local/mimi"
SUPPORTED_LANGUAGES = ["en"]
