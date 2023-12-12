from os.path import expanduser

VICUNA_PATH = expanduser("~/lijun_models/vicuna-13b-v1.5-16k")  # ADD PATH
LLAMA_PATH = expanduser("~/models/llama-2-7b-chat-hf")  # ADD PATH

VICUNA_API_LINK = "https://..."  # ADD LINK
LLAMA_API_LINK = "https://..."  # ADD LINK


ATTACK_TEMP = 1
TARGET_TEMP = 0
ATTACK_TOP_P = 0.9
TARGET_TOP_P = 1

# Increase the above allow more streams in parallel
# Decrease it to reduce the memory requirement 
MAX_PARALLEL_STREAMS = 5
