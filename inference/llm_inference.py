import modal
import modal.gpu
import sys
import torch
import pandas as pd
import os
import time
import numpy as np
import json
import re
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
GPU_CONFIG = modal.gpu.A100(count=1, memory=80)

COT_PROMPT = """Context: The study aims to determine if physiological, rhythmic fluctuations of vagal baroreflex gain, which are crucial for maintaining cardiovascular stability, persist during various phases including exercise, post-exercise ischaemia, and recovery.
Step 1: Define what vagal baroreflex gain is and its importance in cardiovascular regulation.
Step 2: Analyze how exercise and ischaemia might influence these baroreflex rhythms.
Step 3: Reflect on the evidence provided about the persistence of these rhythms during the specified conditions.
Answer: Yes. The study clearly demonstrates that moderate exercise leads to an improvement in baroreflex sensitivity among hypertensive patients, which is crucial for their cardiovascular health.
Answer the following question with a similar structure. Your answer should be 'Yes', 'No' or 'Maybe' followed by a justification that directly references the context.
"""
FEW_SHOT_PROMPT = """ Answer the question using the context below.
Context: During a study on cardiovascular responses, researchers found that moderate exercise improves baroreflex sensitivity and reduces blood pressure in hypertensive patients.
Question: Does moderate exercise improve baroreflex sensitivity in hypertensive patients?
Answer: Yes. The study clearly demonstrates that moderate exercise leads to an improvement in baroreflex sensitivity among hypertensive patients, which is crucial for their cardiovascular health.
Answer the following question with a similar structure. Your answer should be 'Yes', 'No' or 'Maybe' followed by a justification that directly references the context.
"""
DETAILED_PROMPT = """Please answer the question based solely on the context provided, without inferring or adding information not present in the context. Your answer should be 'Yes' or 'No' or 'Maybe' followed by a justification that directly references the context. Output it in the following format.{ Decision}. {Explanation}"""
BASIC_PROMPT = """Please answer the question based solely on the context provided. Your answer should be 'Yes', 'No' or 'Maybe' followed by a justification that directly references the context."""




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
        "vllm==0.4.1",
        "torch==2.2.1",
        "transformers==4.40.1",
        "ray==2.10.0",
        "hf-transfer==0.1.6",
        "huggingface_hub==0.22.2",
        "pandas==2.2.2",
        "packaging==24.0",
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
        secrets=[modal.Secret.from_name("huggingface-secret"), modal.Secret.from_name("wandb-secret")],
    )
)

app = modal.App(
    "medAlpaca-Inference-Final"
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
        self.template = "{user}"

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
            max_tokens=512,
            repetition_penalty=1.1,
            logprobs=0,
            truncate_prompt_tokens=512,
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
            # text_delta = output.outputs[0].text[index:]
            # index = len(output.outputs[0].text)
            num_tokens = len(output.outputs[0].token_ids)
            log_prob = [[(elem.logprob, elem.decoded_token) 
                         for elem in x.values()][0]
                        for x in output.outputs[0].logprobs]

            yield (log_prob, num_tokens)
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

def extract_first_alpha(input_string):
    match = re.search(r"^[A-Za-z]+", input_string)
    if match:
        return match.group()
    return ""

def truncate_conversation_history(tokenizer, conversation_history, new_response, max_sequence_length):
    """
    Truncates the conversation history to fit within the maximum sequence length,
    while ensuring that the new response is included.
    """
    # conversation_bytes = (conversation_history + new_response).encode('utf-8')

    # # Truncate the bytes sequence to the maximum sequence length
    # truncated_bytes = conversation_bytes[-max_sequence_length:]

    # # Decode the truncated bytes back to a string
    # truncated_history = truncated_bytes.decode('utf-8', errors='ignore')

    return conversation_history + new_response

def load_test_data(tokenizer, mode):
    queries = []
    count = 0
    max_sequence_length = 510
    with open('data/pubmedqa-labeled.jsonl', mode='r') as infile:
        for line in infile:
            json_line = json.loads(line)
            question = json_line['question']
            context = ' '.join(json_line['context']['contexts'])
            final_decision = json_line['final_decision']
            # print(context)
            full_query = ''
            full_query = full_query + 'Context:' + context + 'Question:' + question
            if mode == 'base':
                full_query = truncate_conversation_history(tokenizer, BASIC_PROMPT, full_query, max_sequence_length)
            elif mode == 'detailed':
                full_query = truncate_conversation_history(tokenizer, DETAILED_PROMPT, full_query, max_sequence_length)
            elif mode == 'cot':
                full_query = truncate_conversation_history(tokenizer, COT_PROMPT, full_query, max_sequence_length)
            else:
                full_query = truncate_conversation_history(tokenizer, FEW_SHOT_PROMPT, full_query, max_sequence_length)
            queries.append((full_query, final_decision))
            count += 1
            # if count > 20:
            #     break
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
    tokenizer = "tokenizer"
    df = pd.DataFrame()
    current_mode = "base"
    queries = load_test_data(tokenizer, current_mode)
    current_mode_list = ["base", "detailed", "cot", "few"]
    for current_mode in current_mode_list:
        for query in queries:
            question = query[0]
            log_prob = None
            prev = None
            for output in model.completion_stream.remote_gen(question):
                prev = log_prob
                log_prob = output
            tokens = prev[1]
            prev = prev[0]
            full_answer = return_sequence(prev)
            new_row = pd.DataFrame({
                'Question': [question[-512:]],
                'Answer': [full_answer],
                'Log_Probability': [calculate_log_prob(prev)],
                'Accuracy': [query[1] in full_answer],
                'Token_Length': [tokens]
            })
            # Use pd.concat to append the new row to the existing DataFrame
            df = pd.concat([df, new_row], ignore_index=True)
            # df = df.append({'Question': query, 'Answer': return_sequence(prev), 'Log_Probability': calculate_log_prob(prev), 'Accuracy': extract_first_alpha(prev) == query[1]}, ignore_index=True)
        df.to_csv(f'{current_mode}.csv')

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
