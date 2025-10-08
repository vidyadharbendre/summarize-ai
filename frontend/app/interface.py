import os
import gradio as gr
import requests

API_URL = os.environ.get("API_URL", "http://backend:8000/summarize")

MODEL_CHOICES = {
    "T5-small": "t5",
    "BART (Facebook)": "bart"
}

# def get_summary(text, model_choice, max_length, min_length):
#     if not text or not model_choice:
#         return "Please enter text and select a model."
#     try:
#         # Pass length parameters to backend
#         response = requests.post(
#             f"{API_URL}?model_type={MODEL_CHOICES[model_choice]}&max_length={max_length}&min_length={min_length}",
#             json={"text": text}, 
#             timeout=60  # Increased timeout for longer summaries
#         )
#         response.raise_for_status()
#         return response.json().get("summary", "No summary found.")
#     except Exception as e:
#         return f"Error: {e}"

def get_summary(text, model_choice, max_length, min_length):
    if not text or not model_choice:
        return "Please enter text and select a model."
    try:
        response = requests.post(
            f"{API_URL}?model_type={MODEL_CHOICES[model_choice]}",
            json={
                "text": text,
                "max_length": int(max_length),
                "min_length": int(min_length)
            }, 
            timeout=60
        )
        response.raise_for_status()
        return response.json().get("summary", "No summary found.")
    except Exception as e:
        return f"Error: {e}"

iface = gr.Interface(
    fn=get_summary,
    inputs=[
        gr.Textbox(lines=8, label="Enter Text to Summarize", placeholder="Paste your text here..."),
        gr.Radio(list(MODEL_CHOICES.keys()), label="Select Model", value="T5-small"),
        gr.Slider(minimum=50, maximum=300, value=150, step=10, label="Max Summary Length"),
        gr.Slider(minimum=10, maximum=100, value=30, step=5, label="Min Summary Length")
    ],
    outputs=gr.Textbox(lines=10, label="Summary", show_copy_button=True),  # Larger output window
    title="OOP & SOLID Text Summarization (FastAPI + Gradio)",
    description="Enter text, select model, and adjust summary length. FastAPI backend with Hugging Face models.",
    theme=gr.themes.Soft()  # Nice theme
)

if __name__ == "__main__":
    # Keep 0.0.0.0 for Docker compatibility
    iface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
