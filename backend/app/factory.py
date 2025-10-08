from .summarizer import T5Summarizer, BartSummarizer, PegasusSummarizer, DistilBartSummarizer

class SummarizerFactory:
    @staticmethod
    def get_summarizer(model_type: str):
        if model_type == "t5":
            return T5Summarizer()
        elif model_type == "bart":
            return BartSummarizer()
        elif model_type == "pegasus":
            return PegasusSummarizer()
        elif model_type == "distilbart":
            return DistilBartSummarizer()
        else:
            raise ValueError("Unsupported summarizer type")
