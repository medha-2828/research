
import asyncio
import csv
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI

# ==================================================
# CONFIGURATION — EDIT HERE
# ==================================================

BASE_URL = "http://localhost:1234/v1"
API_KEY = "lm-studio"

MODEL = "google/gemma-4-e2b"  # Empty = use the first model returned by LM Studio

SYSTEM_PROMPT = """
Answer the user's question briefly and accurately.
""".strip()

INPUT_FILE = Path("test1.csv")
OUTPUT_FILE = Path("results1.csv")

QUESTION_COLUMN = "question"
MAX_CONCURRENT = 1

# ==================================================
# MAIN
# ==================================================

async def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    # Load questions from CSV.
    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

    if QUESTION_COLUMN not in df.columns:
        raise ValueError(
            f"CSV must contain a column named '{QUESTION_COLUMN}'. "
            f"Found columns: {list(df.columns)}"
        )

    questions = df[QUESTION_COLUMN].dropna().astype(str).tolist()

    if not questions:
        raise ValueError("No questions found in the input CSV.")

    client = AsyncOpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
    )

    try:
        # Choose model.
        if MODEL:
            model = MODEL
        else:
            models = await client.models.list()

            if not models.data:
                raise RuntimeError(
                    "No models found. Load a model in LM Studio."
                )

            model = models.data[0].id

        print(f"Model: {model}")
        print(f"Questions: {len(questions)}")

        semaphore = asyncio.Semaphore(MAX_CONCURRENT)

        async def process_question(index, question):
            async with semaphore:
                try:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=[
                            {
                                "role": "system",
                                "content": SYSTEM_PROMPT,
                            },
                            {
                                "role": "user",
                                "content": question,
                            },
                        ],
                        # No explicit token limit.
                    )

                    return {
                        "index": index,
                        "model": model,
                        "system_prompt": SYSTEM_PROMPT,
                        "equsion": question,
                        "answer": (
                            response.choices[0].message.content or ""
                        ),
                        "error": "",
                    }

                except Exception as exc:
                    return {
                        "index": index,
                        "model": model,
                        "system_prompt": SYSTEM_PROMPT,
                        "equsion": question,
                        "answer": "",
                        "error": str(exc),
                    }

        # Save each result as it finishes.
        fieldnames = [
            "index",
            "model",
            "system_prompt",
            "equsion",
            "answer",
            "error",
        ]

        tasks = [
            asyncio.create_task(process_question(i, q))
            for i, q in enumerate(questions, start=1)
        ]

        successful = 0
        failed = 0

        with OUTPUT_FILE.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            file.flush()

            for task in asyncio.as_completed(tasks):
                result = await task
                writer.writerow(result)
                file.flush()

                if result["error"]:
                    failed += 1
                    print(f"[{result['index']}] Failed")
                else:
                    successful += 1
                    print(f"[{result['index']}] Completed")

        print("\nBatch completed.")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Output: {OUTPUT_FILE.resolve()}")

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
