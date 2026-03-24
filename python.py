from transformers import T5ForConditionalGeneration, AutoTokenizer

MODEL_DIR = "./flan_t5_gender_inclusive"  # ✅ correct path

model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)

def generate(text):
    inputs = tokenizer(text, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_length=128,
        num_beams=4,
        early_stopping=True
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


print(generate("""Rewrite the following sentence to be gender-inclusive:
Each salesman must submit his report.
Answer:"""))
# 🔥 TEST CASES
print(generate("""Rewrite the following sentence to be gender-inclusive:
The chairman announced his decision.
Answer:"""))

print(generate("""Rewrite the following sentence to be gender-inclusive:
A programmer should write his code carefully.
Answer:"""))

print(generate("""Rewrite the following sentence to be gender-inclusive:
Every student must submit his assignment.
Answer:"""))