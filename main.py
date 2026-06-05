from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoConfig, pipeline

# To load: fastapi dev main.py  | or |  fastapi run main.py
# In second terminal: streamlit run app.py
# To stop: ctrl + C

MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    "fine_tuned": {
        "hf_id": "lindybujak/airline-review-modified",
        "display_name": "Fine-tuned airline review model",
    },
    "pretrained_base": {
        "hf_id": "distilbert-base-uncased",
        "display_name": "Pretrained DistilBERT base model",
    },
    "sentiment_pretrained": {
        "hf_id": "distilbert-base-uncased-finetuned-sst-2-english",
        "display_name": "Prebuilt sentiment model (SST-2)",
    },
}
DEFAULT_MODEL_KEY = "fine_tuned"
model_pipelines: Dict[str, Any] = {}
model_configs: Dict[str, Dict[str, Any]] = {}


def validate_model_name(model_name: str) -> str:
    if model_name not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown model_name '{model_name}'. "
                f"Valid options are: {', '.join(MODEL_REGISTRY.keys())}."
            ),
        )
    return model_name


def load_model(model_key: str) -> None:
    model_id = MODEL_REGISTRY[model_key]["hf_id"]

    try:
        print(f"Loading model {model_key} ({model_id})... This may take a few seconds.")
        sentiment_analyzer = pipeline(
            "text-classification",
            model=model_id,
            tokenizer=model_id,
        )
        config = AutoConfig.from_pretrained(model_id).to_dict()
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{model_id}'. Error: {str(e)}")

    model_pipelines[model_key] = sentiment_analyzer
    model_configs[model_key] = config


# Initialize API
app = FastAPI(
    title="Airline Sentiment Analysis API",
    description="A REST API serving multiple transformer sentiment models.",
    version="1.0.0",
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema (what incoming data should look like)
class ReviewRequest(BaseModel):
    review_text: str


@app.on_event("startup")
async def startup_event() -> None:
    load_model(DEFAULT_MODEL_KEY)


# Endpoint
@app.post("/predict/sentiment")
async def predict_sentiment(
    request: ReviewRequest,
    model_name: str = DEFAULT_MODEL_KEY,
):
    model_key = validate_model_name(model_name)

    if len(request.review_text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Review text cannot be empty.")

    if model_key not in model_pipelines:
        load_model(model_key)

    sentiment_analyzer = model_pipelines[model_key]
    result = sentiment_analyzer(request.review_text)[0]

    sentiment_label = result["label"]
    confidence = round(result["score"] * 100, 2)

    return {
        "status": "success",
        "input_text": request.review_text,
        "prediction": sentiment_label,
        "confidence_score": f"{confidence}%",
        "selected_model": MODEL_REGISTRY[model_key]["display_name"],
        "model_name": model_key,
    }


# Get method for model details
@app.get("/model/info")
async def read_model_info(model_name: str = DEFAULT_MODEL_KEY):
    model_key = validate_model_name(model_name)

    if model_key not in model_configs:
        load_model(model_key)

    return {
        "status": 200,
        "model_name": model_key,
        "selected_model": MODEL_REGISTRY[model_key]["display_name"],
        "data": model_configs[model_key],
    }
