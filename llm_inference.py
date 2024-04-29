import modal
import modal.gpu
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import os
import time
import numpy as np
import json

# # Define the model and tokenizer directories
# model_directory = "/Users/karansampath/.cache/lm-studio/models/TheBloke/medalpaca-13B-GGUF"
# tokenizer_directory = "/path/to/your/local/medalpaca/tokenizer"

# # Load the tokenizer and model from the local directory
# tokenizer = AutoTokenizer.from_pretrained(tokenizer_directory)
# model = AutoModelForCausalLM.from_pretrained(model_directory)

# # Define your Modal app
# app = modal.App("local-model-inference")


MODEL_DIR = "/model"
MODEL_NAME = "TheBloke/medalpaca-13B-GPTQ"
MODEL_REVISION = "main"
GPU_CONFIG = modal.gpu.A10G(count=1)


def download_model_to_image(model_dir, model_name, model_revision):
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(model_dir, exist_ok=True)

    snapshot_download(
        model_name,
        revision=model_revision,
        local_dir=model_dir,
        ignore_patterns=["*.pt", "*.bin"],  # Using safetensors
    )
    move_cache()

def get_data(dataset):
    test_df = pd.read_parquet('pubmedqa-labeled.parquet')


vllm_image = (
    modal.Image.debian_slim()
    .pip_install(
        "vllm==0.4.0.post1",
        "torch==2.1.2",
        "transformers==4.39.3",
        "ray==2.10.0",
        "hf-transfer==0.1.6",
        "huggingface_hub==0.22.2",
        "pandas==2.2.2"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model_to_image,
        timeout=60 * 20,
        kwargs={
            "model_dir": MODEL_DIR,
            "model_name": MODEL_NAME,
            "model_revision": MODEL_REVISION,
        },
        secrets=[modal.Secret.from_name("my-huggingface-secret"), modal.Secret.from_name("wandb-secret")],
    )
)

app = modal.App(
    "medAlpaca-Inference"
)  # Note: prior to April 2024, "app" was called "stub"



@app.cls(
    gpu=GPU_CONFIG,
    timeout=60 * 10,
    container_idle_timeout=60 * 10,
    allow_concurrent_inputs=10,
    image=vllm_image,
)
class Model:
    @modal.enter()
    def start_engine(self):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine

        print("🥶 cold starting inference")
        start = time.monotonic_ns()

        engine_args = AsyncEngineArgs(
            model=MODEL_DIR,
            tensor_parallel_size=GPU_CONFIG.count,
            gpu_memory_utilization=0.90,
            enforce_eager=False,  # capture the graph for faster inference, but slower cold starts
            disable_log_stats=True,  # disable logging so we can stream tokens
            disable_log_requests=True,
        )
        self.template = "<s> [INST] {user} [/INST] "

        # this can take some time!
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        duration_s = (time.monotonic_ns() - start) / 1e9
        print(f"🏎️ engine started in {duration_s:.0f}s")

    @modal.method()
    async def completion_stream(self, user_question):
        from vllm import SamplingParams
        from vllm.utils import random_uuid

        sampling_params = SamplingParams(
            temperature=0.75,
            max_tokens=128,
            repetition_penalty=1.1,
            logprobs=0,
        )

        request_id = random_uuid()
        result_generator = self.engine.generate(
            self.template.format(user=user_question),
            sampling_params,
            request_id,
        )
        index, num_tokens = 0, 0
        start = time.monotonic_ns()
        async for output in result_generator:
            if (
                output.outputs[0].text
                and "\ufffd" == output.outputs[0].text[-1]
            ):
                continue
            text_delta = output.outputs[0].text[index:]
            index = len(output.outputs[0].text)
            num_tokens = len(output.outputs[0].token_ids)
            log_prob = [[(elem.logprob, elem.decoded_token) 
                         for elem in x.values()][0]
                        for x in output.outputs[0].logprobs]

            yield log_prob
        duration_s = (time.monotonic_ns() - start) / 1e9

        yield (
            f"\n\tGenerated {num_tokens} tokens from {MODEL_NAME} in {duration_s:.1f}s,"
            f" throughput = {num_tokens / duration_s:.0f} tokens/second on {GPU_CONFIG}.\n"
        )

    @modal.exit()
    def stop_engine(self):
        if GPU_CONFIG.count > 1:
            import ray

            ray.shutdown()


def calculate_log_prob(log_prob):
    return np.mean([x[0] for x in log_prob])


def return_sequence(log_prob):
    return ''.join([x[1] for x in log_prob])


def load_test_data():
    queries = []
    with open('data/pubmedqa-labeled.jsonl', mode='r') as infile:
        for line in infile:
            json_line = json.loads(line)
            question = json_line['question']
            context = ' '.join(json_line['context']['contexts'])
            final_decision = json_line['final_decision']
            # print(context)
            full_query = ' '.join(['Context:', context, '\n',
                                   'Question:', question, ])
            queries.append((full_query, final_decision))
    return queries


@app.local_entrypoint()
def main():
    # questions = [
    #     "What are the common symptoms and treatments for Type 2 Diabetes?",
    #     # "How does the influenza virus transmit from one person to another?",
    #     # "What are the latest advancements in cancer immunotherapy?",
    #     # "Describe the role of telemedicine in improving healthcare accessibility.",
    #     # "What is the normal range for blood pressure in adults?",
    #     # "Who was Hippocrates and what is his significance in the history of medicine?",
    #     ]
    model = Model()
    queries = load_test_data()
    for query in queries:
        question = query[0]
        log_prob = None
        prev = None
        for output in model.completion_stream.remote_gen(question):
            prev = log_prob 
            log_prob = output
        print(calculate_log_prob(prev))
        print(return_sequence(prev))





# @app.function()
# def run_llm(question, context):
#     # Combine the context and question into a single input text
#     input_text = f"{context} {question}"
    
#     # Tokenize the input text
#     inputs = tokenizer.encode(input_text, return_tensors="pt")
    
#     # Generate a response from the model
#     outputs = model.generate(inputs, max_length=512)
    
#     # Decode the generated tokens to a string
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return response

# @app.local_entrypoint()
# def main():
#     question = "What is the capital of France?"
#     context = "The capital of France is a city known for its history and culture."
#     result = run_llm.remote(question, context)
#     print("LLM output:", result)

# # Deploy the app
# if __name__ == "__main__":
#     app.run()
