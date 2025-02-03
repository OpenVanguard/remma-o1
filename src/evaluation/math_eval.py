from datasets import load_dataset

def evaluate_math(model):
    # GSM8K Accuracy
    gsm8k = load_dataset("gsm8k", "main")
    correct = 0
    for example in gsm8k["test"]:
        output = model.generate(example["question"])
        if example["answer"].split("####")[-1].strip() in output:
            correct += 1
    print(f"GSM8K Accuracy: {correct/len(gsm8k['test']):.2%}")

    # MATH Category-wise Performance
    math = load_dataset("competition_math")
    # ... similar implementation ...