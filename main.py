from transformers import *
from termcolor import colored, cprint

cprint("-----Branden's AI Frontend-----", "cyan")

# Initial AI Setup
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", device_map="auto", offload_folder="offload")
config = GenerationConfig.from_pretrained("google/flan-t5-xxl")

# Prompt user for a query and config options to feed to the AI
cprint("--------------------------------", "red")
input_text = input("Please state your query: ")
input_max_tokens = input("Please specify the maximum length of the generation: ")
input_min_tokens = input("Please specify the minimum length of the generation: ")
cprint("--------------------------------", "red")

# Config
config.max_new_tokens=int(input_max_tokens)
config.min_new_tokens=int(input_min_tokens)
cprint("[+] Config updated!", "cyan")

# Generate the tokenizer with NVIDIA CUDA
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
cprint("[+] Tokenizer created!", "cyan")

# Generate the prompt
cprint(f"Beginning generation with prompt \"{input_text}\"...", "cyan")
outputs = model.generate(input_ids, generation_config=config)
decoded_output = tokenizer.decode(outputs[0])
final_output = str(decoded_output).replace("<pad>", "").replace("</s>", "")
cprint("Generation complete! The AI's text is shown below:")
print(f"\"{final_output}\"")