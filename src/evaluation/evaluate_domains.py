# src/evaluation/evaluate_domains.py
from tokenizers import Tokenizer
from datasets import load_dataset

def evaluate_model(model):
    # General language evaluation
    wikitext = load_dataset("wikitext", "wikitext-103-v1")
    general_ppl = calculate_perplexity(model, wikitext["test"])
    
    # Math evaluation
    math_test = load_dataset("gsm8k", "main", split="test")
    math_acc = calculate_math_accuracy(model, math_test)
    
    print(f"General PPL: {general_ppl:.2f}")
    print(f"Math Accuracy: {math_acc:.2%}")

def calculate_perplexity(model, dataset):
    # Implement perplexity calculation
    pass

def calculate_math_accuracy(model, dataset):
    # Implement math problem solving accuracy
    pass