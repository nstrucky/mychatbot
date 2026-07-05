
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/blenderbot-400M-distill"

# Load model (download on first run and reference local installation for subsequent runs)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load tokenizer which optimizes input and passes it to the model efficiently
# Meaning it turns input into tokens
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Storing conversation in simple list
conversation_history = []


print("Chatbot is ready! (type exit to quit)\n")

while True:

	# retains only the last few entries to avoid confusion
	conversation_history = conversation_history[:-6]

	history_string = "\n".join(conversation_history)

	input_text = input(">>> ")

	if input_text.lower() == "exit":
		break

	prompt = history_string + f"\nUser: {input_text}\nBot:"

	# Tokenizer converts prompt string into tokens/vector vaules
	# which the model can understand. 
	# return_tensor is PyTorch, truncate cuts off inputs if it exceeds
	# model limits, and max_length limits the number of tokens allowed
	# as inputs
	inputs = tokenizer(
		prompt,
		return_tensors="pt",
		truncation=True,
		max_length=512
	)
	
	# Generate the model's response based on inputs. This output will
	# still be in the form of tokens (numerical values)
	outputs = model.generate(
        **inputs,
        max_new_tokens=60,
        no_repeat_ngram_size=3,
        repetition_penalty=1.3,
        do_sample=True,
        temperature=0.6,
        top_p=0.85
    )


	response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
	print(f"Bot: {response}")

	conversation_history.append(f"User: {input_text}")
	conversation_history.append(f"Bot: {response}")