from transformers import *

# Initial AI Setup
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", device_map="auto", offload_folder="offload")
config = GenerationConfig.from_pretrained("google/flan-t5-xxl")

# Prompt user for a query and config options to feed to the AI
input_text = input("Please state your query: ")
input_max_tokens = input("Please specify the maximum length of the generation: ")
input_min_tokens = input("Please specify the minimum length of the generation: ")

# Config
config.max_new_tokens=int(input_max_tokens)
config.min_new_tokens=int(input_min_tokens)

# Generate the tokenizer with NVIDIA CUDA
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

# Generate the prompt
outputs = model.generate(input_ids, generation_config=config)
print(tokenizer.decode(outputs[0]))