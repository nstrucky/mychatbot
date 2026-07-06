from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import warnings

warnings.filterwarnings("ignore")

model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"

print("Loading model...")


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
		"content": "You are a helpful AI assistant. Give short and concise answers in 2-3 lines."
	}
]

print("Chatbot started. Type 'exit to quit.\n")

while True:
	user_input = input(">>>")

	if user_input.lower() == "exit":
		break

	messages.append({"role": "user", "content": user_input})

	# To avoid long conversations keep only recent exchanges
	messages = messages[-10:]

	tokenized = tokenizer.apply_chat_template(
    	messages,
    	tokenize=True,
    	add_generation_prompt=True,
    	return_tensors="pt",
    	return_dict=True,
    	max_length=512
	)

	 with torch.inference_mode():
     	outputs = model.generate(
        	tokenized["input_ids"],
        	attention_mask=tokenized["attention_mask"],
        	max_new_tokens=60,
        	temperature=0.5,
        	top_p=0.8,
        	do_sample=True,
        	repetition_penalty=1.3,
        	no_repeat_ngram_size=3,
        	pad_token_id=tokenizer.pad_token_id
    	)

