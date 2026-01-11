import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline

MODEL_LIST = {
    "Qwen 0.5B (Fastest)": "Qwen/Qwen2.5-0.5B-Instruct",
    "TinyLlama 1.1B": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "DistilGPT-2 (Ultra light)": "distilgpt2"
}

def load_llm(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=180,
        temperature=0.7,
        do_sample=True
    )

    return HuggingFacePipeline(pipeline=pipe)
