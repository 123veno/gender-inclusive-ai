from fastapi import FastAPI
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, AutoTokenizer
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],   # 🔥 IMPORTANT
    allow_headers=["*"],   # 🔥 IMPORTANT
)

MODEL_DIR = "./flan_t5_gender_inclusive"

model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)


class Request(BaseModel):
    text: str


def rewrite(text):
    prompt = f"""Rewrite the following sentence to be gender-inclusive:
{text}
Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_length=128,
        num_beams=4,
        early_stopping=True
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.get("/")
def home():
    return {"message": "Gender Inclusive Rewriter API is running"}
@app.post("/rewrite")
def rewrite_api(req: Request):
    result = rewrite(req.text)
    return {"result": result}