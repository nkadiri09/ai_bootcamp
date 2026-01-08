# python
import warnings
warnings.filterwarnings('ignore')
import torch
from transformers import pipeline

device = 0 if torch.cuda.is_available() else -1  # 0 = first GPU, -1 = CPU
generator = pipeline('text-generation', model='gpt2', device=device)

output = generator("Hello, my name is", max_new_tokens=50, do_sample=True, top_k=50)
print(output[0]['generated_text'])
