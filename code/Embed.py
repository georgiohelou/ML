# Import generic wrappers
from transformers import AutoModel, AutoTokenizer 


# Define the model repo
model_name = "distilbert-base-uncased" 


# Download pytorch model
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Transform input tokens 
inputs = tokenizer("Hello world!", return_tensors="pt")

# Model apply
outputs = model(**inputs)
my_list = []
list = outputs
print(list)