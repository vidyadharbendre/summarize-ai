from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from .base import BaseSummarizer
import logging

logger = logging.getLogger(__name__)

class T5Summarizer(BaseSummarizer):
    def __init__(self, model_name: str = "t5-small"):
        # Use manual model loading for better control
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def summarize(self, text: str, max_length: int = 100, min_length: int = 30) -> str:
        if not text.strip():
            return "Please enter some text."
        
        # Add T5 prefix for summarization
        input_text = f"summarize: {text}"
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text, 
            max_length=512, 
            truncation=True, 
            return_tensors="pt"
        )
        
        logger.info(f"T5 - Requested max: {max_length}, min: {min_length}")
        
        try:
            # Use generate() method with explicit parameters
            summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,        # This should work now
                min_length=min_length,        # This should work now
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
            
            # Decode summary
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            word_count = len(summary.split())
            logger.info(f"T5 generated {word_count} words, {len(summary_ids[0])} tokens")
            
            return summary
            
        except Exception as e:
            logger.error(f"T5 error: {e}")
            return f"T5 Error: {e}"

class BartSummarizer(BaseSummarizer):
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        # Use manual model loading for better control
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def summarize(self, text: str, max_length: int = 100, min_length: int = 30) -> str:
        if not text.strip():
            return "Please enter some text."
            
        # Tokenize input
        inputs = self.tokenizer(
            text, 
            max_length=1024, 
            truncation=True, 
            return_tensors="pt"
        )
        
        logger.info(f"BART - Requested max: {max_length}, min: {min_length}")
        
        try:
            # Use generate() method with explicit parameters
            summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,        # This should work now
                min_length=min_length,        # This should work now
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
            
            # Decode summary
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            word_count = len(summary.split())
            logger.info(f"BART generated {word_count} words, {len(summary_ids[0])} tokens")
            
            return summary
            
        except Exception as e:
            logger.error(f"BART error: {e}")
            return f"BART Error: {e}"

class PegasusSummarizer(BaseSummarizer):
    def __init__(self, model_name: str = "google/pegasus-cnn_dailymail"):
        """PEGASUS model - specifically designed for summarization"""
        self.pipeline = pipeline("summarization", model=model_name)

    def summarize(self, text: str, max_length: int = 100, min_length: int = 30) -> str:
        if not text.strip():
            return "Please enter some text."
            
        logger.info(f"PEGASUS - Requested max: {max_length}, min: {min_length}")
        
        try:
            # PEGASUS works well with these parameters
            summary_list = self.pipeline(
                text, 
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True,
                clean_up_tokenization_spaces=True
            )
            
            result = summary_list[0]['summary_text']
            word_count = len(result.split())
            logger.info(f"PEGASUS generated {word_count} words")
            
            return result
            
        except Exception as e:
            logger.error(f"PEGASUS error: {e}")
            return f"PEGASUS Error: {e}"

class DistilBartSummarizer(BaseSummarizer):
    def __init__(self, model_name: str = "sshleifer/distilbart-cnn-12-6"):
        """DistilBART - Smaller, faster version of BART"""
        self.pipeline = pipeline("summarization", model=model_name)

    def summarize(self, text: str, max_length: int = 100, min_length: int = 30) -> str:
        if not text.strip():
            return "Please enter some text."
            
        logger.info(f"DistilBART - Requested max: {max_length}, min: {min_length}")
        
        try:
            summary_list = self.pipeline(
                text, 
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True,
                clean_up_tokenization_spaces=True
            )
            
            result = summary_list[0]['summary_text']
            word_count = len(result.split())
            logger.info(f"DistilBART generated {word_count} words")
            
            return result
            
        except Exception as e:
            logger.error(f"DistilBART error: {e}")
            return f"DistilBART Error: {e}"
