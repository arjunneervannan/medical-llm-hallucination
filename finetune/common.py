from pathlib import PurePosixPath

from modal import App, Image, Volume, Secret

APP_NAME = "axolotl"

# Axolotl image hash corresponding to 0.4.0 release (2024-02-14)
AXOLOTL_REGISTRY_SHA = (
    "d5b941ba2293534c01c23202c8fc459fd2a169871fa5e6c45cb00f363d474b6a"
)

axolotl_image = (
    Image.from_registry(f"winglian/axolotl@sha256:{AXOLOTL_REGISTRY_SHA}")
    .run_commands(
        "git clone https://github.com/OpenAccess-AI-Collective/axolotl /root/axolotl",
        "cd /root/axolotl && git checkout v0.4.0",
    )
    .pip_install("huggingface_hub==0.20.3", "hf-transfer==0.1.5", "wandb==0.16.3", "git+https://github.com/PanQiWei/AutoGPTQ.git@v0.7.1")
    .env(
        dict(
            HUGGINGFACE_HUB_CACHE="/pretrained",
            HF_HUB_ENABLE_HF_TRANSFER="1",
            TQDM_DISABLE="true",
        )
    )
)

vllm_image = Image.from_registry(
    "nvidia/cuda:12.1.0-base-ubuntu22.04", add_python="3.10"
).pip_install(
    "vllm==0.2.6",
    "torch==2.1.2",
    "auto-gptq==0.2.0",
)

app = App(APP_NAME,secrets=[Secret.from_name("huggingface-secret"), Secret.from_name("wandb-secret")])

pretrained_volume = Volume.from_name("example-pretrained-vol", create_if_missing=True)
runs_volume = Volume.from_name("example-runs-vol", create_if_missing=True)
VOLUME_CONFIG: dict[str | PurePosixPath, Volume] = {
    "/pretrained": pretrained_volume,
    "/runs": runs_volume,
}