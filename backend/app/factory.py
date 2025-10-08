from .summarizer import T5Summarizer, BartSummarizer

class SummarizerFactory:
    @staticmethod
    def get_summarizer(model_type: str):
        if model_type == "t5":
            return T5Summarizer()
        elif model_type == "bart":
            return BartSummarizer()
        else:
            raise ValueError(f"Unknown model type: {model_type}")