from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

generator = pipeline('text-generation', model='meta-llama/Llama-3.2-1B')