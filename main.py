from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import spacy

# Initialize FastAPI app
app = FastAPI()

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Load saved ML model + vectorizer
model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# Cleaning function (same as Colab)
def spacy_clean_text(text):
    doc = nlp(text.lower())
    cleaned_tokens = [
        token.lemma_
        for token in doc
        if token.is_alpha and not token.is_stop
    ]
    return " ".join(cleaned_tokens)

# Input schema
class InputText(BaseModel):
    text: str

@app.get("/")
def root():
    return {"status": "Text Classification API running"}

@app.post("/predict")
def predict(data: InputText):
    text = data.text

    # clean text using spaCy
    cleaned = spacy_clean_text(text)

    # convert to vector
    vector = vectorizer.transform([cleaned])

    # make prediction
    pred = model.predict(vector)[0]

    # label mapping
    label_map = {
        0: "Normal",
        1: "Daydreaming",
        2: "Depressive"
    }

    return {
        "input": text,
        "cleaned": cleaned,
        "prediction": label_map[pred]
    }
