from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from .base import BaseSummarizer
import logging

logger = logging.getLogger(__name__)

class T5Summarizer(BaseSummarizer):
    def __init__(self, model_name: str = "t5-small"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"T5Summarizer using device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"CUDA GPU available: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("CUDA GPU not available, using CPU")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def summarize(self, text: str, max_length: int = 100, min_length: int = 30) -> str:
        if not text.strip():
            return "Please enter some text."

        input_text = f"summarize: {text}"
        inputs = self.tokenizer(
            input_text,
            max_length=512,
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        logger.info(f"T5 - Requested max: {max_length}, min: {min_length}")

        try:
            summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                min_length=min_length,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            word_count = len(summary.split())
            logger.info(f"T5 generated {word_count} words, {len(summary_ids[0])} tokens")
            return summary

        except Exception as e:
            logger.error(f"T5 error: {e}")
            return f"T5 Error: {e}"


class BartSummarizer(BaseSummarizer):
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"BartSummarizer using device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"CUDA GPU available: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("CUDA GPU not available, using CPU")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def summarize(self, text: str, max_length: int = 100, min_length: int = 30) -> str:
        if not text.strip():
            return "Please enter some text."

        inputs = self.tokenizer(
            text,
            max_length=1024,
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        logger.info(f"BART - Requested max: {max_length}, min: {min_length}")

        try:
            summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                min_length=min_length,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            word_count = len(summary.split())
            logger.info(f"BART generated {word_count} words, {len(summary_ids[0])} tokens")
            return summary

        except Exception as e:
            logger.error(f"BART error: {e}")
            return f"BART Error: {e}"


class PegasusSummarizer(BaseSummarizer):
    def __init__(self, model_name: str = "google/pegasus-cnn_dailymail"):
        device = 0 if torch.cuda.is_available() else -1
        logger.info(f"PegasusSummarizer pipeline device: {device}")
        if torch.cuda.is_available():
            logger.info(f"CUDA GPU available: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("CUDA GPU not available, using CPU")
        self.pipeline = pipeline("summarization", model=model_name, device=device)

    def summarize(self, text: str, max_length: int = 100, min_length: int = 30) -> str:
        if not text.strip():
            return "Please enter some text."

        logger.info(f"PEGASUS - Requested max: {max_length}, min: {min_length}")

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
            logger.info(f"PEGASUS generated {word_count} words")
            return result

        except Exception as e:
            logger.error(f"PEGASUS error: {e}")
            return f"PEGASUS Error: {e}"


class DistilBartSummarizer(BaseSummarizer):
    def __init__(self, model_name: str = "sshleifer/distilbart-cnn-12-6"):
        device = 0 if torch.cuda.is_available() else -1
        logger.info(f"DistilBartSummarizer pipeline device: {device}")
        if torch.cuda.is_available():
            logger.info(f"CUDA GPU available: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("CUDA GPU not available, using CPU")
        self.pipeline = pipeline("summarization", model=model_name, device=device)

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
