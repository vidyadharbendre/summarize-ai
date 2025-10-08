from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from .factory import SummarizerFactory
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
summarizer_instances = {
    "t5": SummarizerFactory.get_summarizer("t5"),
    "bart": SummarizerFactory.get_summarizer("bart"),
    "pegasus": SummarizerFactory.get_summarizer("pegasus"),
    "distilbart": SummarizerFactory.get_summarizer("distilbart")
}

class TextRequest(BaseModel):
    text: str
    max_length: int = 150
    min_length: int = 30

class SummaryResponse(BaseModel):
    summary: str

@app.post("/summarize", response_model=SummaryResponse)
async def summarize(
    request: TextRequest,
    model_type: str = Query("t5", enum=["t5", "bart", "pegasus", "distilbart"])
):
    # EXPLICIT DEBUGGING - This should show in logs
    print(f"🔍 DEBUG: Received request with max_length={request.max_length}, min_length={request.min_length}")
    logger.info(f"🔍 API received: model={model_type}, max={request.max_length}, min={request.min_length}")
    
    try:
        summarizer = summarizer_instances.get(model_type)
        if summarizer is None:
            raise HTTPException(status_code=400, detail="Invalid model_type requested.")
        
        # Call summarizer with explicit logging
        logger.info(f"🔍 Calling summarizer.summarize with max={request.max_length}, min={request.min_length}")
        result = summarizer.summarize(
            request.text, 
            max_length=request.max_length, 
            min_length=request.min_length
        )
        
        # Log result length
        word_count = len(result.split())
        logger.info(f"🔍 Result: {word_count} words generated")
        
        return SummaryResponse(summary=result)
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}
