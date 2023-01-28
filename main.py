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
input_temp =  input("Please specify the temperature for the AI: ")
cprint("--------------------------------", "red")

# Config
config.max_new_tokens=int(input_max_tokens)
config.min_new_tokens=int(input_min_tokens)
config.temperature=int(input_temp)
cprint("[+] Config updated!", "cyan")

# Generate the tokenizer with NVIDIA CUDA
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
cprint("[+] Tokenizer created!", "cyan")

# Generate the prompt
colored_input = colored(input_text, "yellow") # Can probably be optimized somehow...
cprint(f"Beginning generation with prompt \"{colored_input}\"...", "cyan")
outputs = model.generate(input_ids, generation_config=config) # Actual generation function. Defaults to PyTorch.
decoded_output = tokenizer.decode(outputs[0]) # Decode the generated artifacts
final_output = str(decoded_output).replace("<pad>", "").replace("</s>", "").strip().capitalize() # Remove padding & properly format the string
cprint("Generation complete! The AI's text is shown below:", "cyan")
print(f"\"{final_output}\"")