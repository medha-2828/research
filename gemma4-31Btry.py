
import time
import threading
import torch
import gradio as gr

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)
from transformers.utils import logging

# ============================================================
# CONFIG
# ============================================================

MODEL_ID = "google/gemma-4-E4B-it"
MAX_NEW_TOKENS = 256

# Set True to print detailed Transformers loading information.
VERBOSE = True

if VERBOSE:
    logging.set_verbosity_info()

print("=" * 60)
print(" GEMMA 4 LOCAL CHAT")
print("=" * 60)
print(f"Model: {MODEL_ID}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. Check your PyTorch setup.")

print(f"GPU: {torch.cuda.get_device_name(0)}")
print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("Loading model; this may take a while...")
load_start = time.perf_counter()

torch.set_float32_matmul_precision("high")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="cuda:0",          # Force 100% onto RTX 5090 (no CPU fallback)
    torch_dtype=torch.bfloat16,   # Native Blackwell BF16 tensor cores
    attn_implementation="sdpa",   # Fused attention kernels
    low_cpu_mem_usage=True,
)

model.eval()

print(f"Model loaded in {time.perf_counter() - load_start:.1f}s")
print("\nDevice placement:")
if hasattr(model, "hf_device_map"):
    for name, device in model.hf_device_map.items():
        print(f"  {name}: {device}")
else:
    print(f"  Single device: {model.device}")

print(f"\nGPU memory allocated: "
      f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GiB")
print(f"GPU memory reserved: "
      f"{torch.cuda.memory_reserved(0) / 1024**3:.2f} GiB")
print("=" * 60)


# ============================================================
# CHAT FUNCTION
# ============================================================

def chat(message, history):
    # Build the conversation from previous Gradio messages.
    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant."
        }
    ]

    for item in history:
        role = item.get("role")
        content = item.get("content")

        # This demo supports text messages only.
        if role in ("user", "assistant") and isinstance(content, str):
            messages.append({
                "role": role,
                "content": content
            })

    messages.append({
        "role": "user",
        "content": message
    })

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    # Place input tokens alongside the embedding layer.
    input_device = model.get_input_embeddings().weight.device
    inputs = inputs.to(input_device)

    input_tokens = inputs["input_ids"].shape[1]

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=600,
    )

    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "use_cache": True,
        "cache_implementation": "static",
    }

    print("\n" + "=" * 60)
    print(f"USER: {message}")
    print(f"Prompt tokens: {input_tokens}")
    print(f"Input device: {input_device}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print("Starting generation...")

    start = time.perf_counter()

    def generate():
        with torch.inference_mode():
            model.generate(**generation_kwargs)

    thread = threading.Thread(target=generate, daemon=True)
    thread.start()

    response = ""
    first_token_time = None

    # Stream chunks as soon as the streamer produces them.
    for chunk in streamer:
        if first_token_time is None:
            first_token_time = time.perf_counter()
            print(
                f"First streamed text chunk after "
                f"{first_token_time - start:.2f}s"
            )

        response += chunk
        print(chunk, end="", flush=True)
        yield response

    thread.join()

    elapsed = time.perf_counter() - start
    print("\n\nGeneration complete.")
    print(f"Elapsed time: {elapsed:.2f}s")
    print(f"Input tokens: {input_tokens}")
    print(f"Output tokens (approx.): {len(tokenizer.encode(response))}")

    if torch.cuda.is_available():
        print(
            f"Peak GPU allocated: "
            f"{torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GiB"
        )

    print("=" * 60)


# ============================================================
# GRADIO UI
# ============================================================


demo = gr.ChatInterface(
    fn=chat,
    title="Gemma 4 — Local RTX 5090 Chat",
    description=(
        "Local Hugging Face inference with streaming output. "
        "Watch the terminal for verbose logs."
    ),
)


if __name__ == "__main__":
    print("\nStarting Gradio at http://127.0.0.1:7860")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )
