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

SYSTEM_PROMPT_VERSION = "no_pref_v1"
EXTRACT_PREF_PROMPT_VERSION = "v2"
