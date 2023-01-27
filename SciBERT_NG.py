from transformers import *

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl");
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", device_map="auto", offload_folder="offload");
config = GenerationConfig.from_pretrained("google/flan-t5-xxl");

# Config
config.max_new_tokens=80;
config.min_new_tokens=20;

input_text = input("Please state your query: ");
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda");

outputs = model.generate(input_ids, generation_config=config);
print(tokenizer.decode(outputs[0]));