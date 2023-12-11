VICUNA_PATH = "/nvme/wangruohui/vicuna-7b-v1.5/" # ADD PATH
LLAMA_PATH = "/nvme/wangruohui/Llama-2-7b-chat-hf" # ADD PATH

VICUNA_API_LINK ="http://localhost:8000" # ADD LINK
LLAMA_API_LINK = "http://localhost:8000" # ADD LINK


ATTACK_TEMP = 1
TARGET_TEMP = 0
ATTACK_TOP_P = 0.9
TARGET_TOP_P = 1

# Increase the above allow more streams in parallel
# Decrease it to reduce the memory requirement 
MAX_PARALLEL_STREAMS = 5
