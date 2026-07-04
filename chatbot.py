
from transormers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/blenderbot-400M-distill"

# Load model (download on first run and reference local installation for subsequent runs)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load tokenizer which optimizes input and passes it to the model efficiently
# Meaning it turns input into tokens
tokenizer = AutoTokenizer.from_pretrained(model_name)