# TODO - refactor this file to serve functions for web interface

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import warnings

warnings.filterwarnings("ignore")

model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# In transformer models, inputs in a batch must often be the same length.
# Shorter sequences are padded with a special token called the padding 
# token (pad_token). This tells the model which parts of the input are real
# words and which are filler.
tokenizer.pad_token = tokenizer.unk_token

model = AutoModelForCausalLM.from_pretrained(
	model_name,
	device_map="cpu", # Controls where the model runs (CPU or GPU)
	torch_dtype=torch.float32 # sets numerical precision for balancing performance
)

# In modern chat-based LLMs, we use a structured conversation format made
# of messages. Each message has a specific role that tells the model who
# is speaking and how to behave.

# Messages are the full history between user and AI
messages = [
	{
		"role": "system",
		"content": "You are a an old prospector from Utah who answers questions helpfull, but in a very western style."
	}
]

while True:
	user_input = input(">>>")

	if user_input.lower() == "exit":
		break

	# Note the role user (i.e. can either be user, system, or assistant)
	messages.append({"role": "user", "content": user_input})

	# To avoid long conversations keep only recent exchanges
	messages = messages[-10:]

	# Generate tokenized version of messages list. This also conforms the
	# input into a conversation template this model will use (instead
	# of generating the conversations manually like in chatbot.py)
	tokenized = tokenizer.apply_chat_template(
    	messages,
    	tokenize=True,
    	add_generation_prompt=True,
    	return_tensors="pt",
    	return_dict=True,
    	max_length=512
	)

	# Generate the model's response, in inference mode (no training)
	# This mode makes the generation faster and memory-efficient
	with torch.inference_mode():
		outputs = model.generate(
        	tokenized["input_ids"],
        	attention_mask=tokenized["attention_mask"],
        	max_new_tokens=120,
        	temperature=0.9,
        	top_p=0.75,
        	do_sample=True,
        	repetition_penalty=1.3,
        	no_repeat_ngram_size=3,
        	pad_token_id=tokenizer.pad_token_id
    	)


    # Decode and display the response
	response = tokenizer.decode(
		# extract only the newly generated response from model
		outputs[0][tokenized["input_ids"].shape[-1]:],
		skip_special_tokens=True
    )

	# Save assistant response (i.e., the model's response)
	messages.append({"role": "assistant", "content": response})