RANDOM_SEED = 26

MODEL_TEMPERATURE = 0.3

USE_BACKEND = "openai"  # "ollama" or "openai"

# if USE_BACKEND == "ollama"
# MODEL_NAME = "llama3.1"
OLLAMA_URL = "https://ollama.kube.isc.heia-fr.ch"

# if USE_BACKEND == "openai": choose one from bellow
# 1
# MODEL_NAME = "llama3.1"
# OPENAI_BASE_URL = "https://ollama.kube.isc.heia-fr.ch/v1"
# OPENAI_API_KEY = "-"

# 2
MODEL_NAME = "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"
# "http://llama3.kube-ext.isc.heia-fr.ch/v1/"
OPENAI_BASE_URL = "http://localhost:8000/v1/"
OPENAI_API_KEY = "-"

# Chatbot modality mode
# If True, visuals are enabled.  If False, you get text-only responses.
USE_IMAGES = True

# Prompt versions
if USE_IMAGES:
    SYSTEM_PROMPT_VERSION = "v3.1"
else:
    SYSTEM_PROMPT_VERSION = "v2"
EXTRACT_PREF_PROMPT_VERSION = "v2.2"
MERGE_PREF_PROMPT_VERSION = "v4.2"  # 4.2 the only working one
PICTURE_AGENT_SEARCH_PROMPT_VERSION = "v2.1"
PICTURE_AGENT_SELECT_PROMPT_VERSION = "v2.1"

# Data path
EXERCISES_CSV_PATH = "dataset/exercises_working_gifs.csv"
