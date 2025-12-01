import ast
import pandas as pd
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
import evaluate

TASK_PREFIX = "translate English to Spanish: "
MAX_LENGTH = 256

bleu = evaluate.load("bleu")
chrf = evaluate.load("chrf")

# Fix corrupted dataset: clean "['text']" → "text"


def clean_list_string(x):
    """
    Converts strings like "['text here']" into "text here".
    If the value is already a string without brackets, return it unchanged.
    """
    if isinstance(x, str) and x.startswith("[") and x.endswith("]"):
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list) and len(parsed) == 1:
                return parsed[0]
        except Exception:
            return x
    return x


# Generation function
def generate_translation(model, tokenizer, device, text):
    input_text = TASK_PREFIX + text
    inputs = tokenizer(input_text, return_tensors="pt",
                       truncation=True).input_ids.to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=MAX_LENGTH,
            num_beams=4,
            early_stopping=True,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Evaluation of a single split
def evaluate_split(model, tokenizer, device, en_list, es_list, split_name="", print_every=25):
    predictions = []
    references = []

    total = len(en_list)
    print(f"\n→ Evaluating {split_name} ({total} samples)...")

    for i, (en, es) in enumerate(zip(en_list, es_list)):
        pred = generate_translation(model, tokenizer, device, en)
        predictions.append(pred)
        references.append([es])

        if (i + 1) % print_every == 0:
            print(f"   {split_name}: {i + 1}/{total} done")

    print(f"✓ Completed {split_name}.\n")
    return {
        "bleu": bleu.compute(predictions=predictions, references=references),
        "chrf": chrf.compute(predictions=predictions, references=references)
    }


# Full evaluation pipeline
def evaluate_all(model_path, df_clean, df_corr):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    model.eval()

    print(f"Loaded model on {device}.")

    results = {}

    clean_en = df_clean["en"].tolist()
    clean_es = df_clean["es"].tolist()

    # Clean
    results["clean"] = evaluate_split(
        model, tokenizer, device,
        clean_en,
        clean_es,
        split_name="CLEAN"
    )

    # Corruption level 1
    results["corruption_1"] = evaluate_split(
        model, tokenizer, device,
        df_corr["en_corruption_1"].tolist(),
        clean_es,
        split_name="CORRUPTION 1"
    )

    # Corruption level 2
    results["corruption_2"] = evaluate_split(
        model, tokenizer, device,
        df_corr["en_corruption_2"].tolist(),
        clean_es,
        split_name="CORRUPTION 2"
    )

    # Corruption level 3
    results["corruption_3"] = evaluate_split(
        model, tokenizer, device,
        df_corr["en_corruption_3"].tolist(),
        clean_es,
        split_name="CORRUPTION 3"
    )

    return results


# Print results
def print_results(results):
    print("\n================ METRIC RESULTS ================\n")

    for key, metrics in results.items():
        print(f"--- {key.upper()} ---")
        print(f"BLEU : {metrics['bleu']['bleu']:.4f}")
        print(f"chrF : {metrics['chrf']['score']:.4f}")
        print()


# Main script — load, clean, align, merge
if __name__ == "__main__":

    df_clean = pd.read_csv("./data/t5_translated.csv")
    df_corr = pd.read_csv("./data/t5_corrupt_translated.csv")

    df_clean["id"] = range(len(df_clean))
    df_corr["id"] = range(len(df_corr))

    for col in ["es_corruption_1", "es_corruption_2", "es_corruption_3"]:
        df_corr[col] = df_corr[col].apply(clean_list_string)

    df = df_corr.merge(
        df_clean[["id", "en", "es"]],
        on="id",
        how="inner"
    )

    print(f"Merged dataset size: {len(df)} rows")

    results = evaluate_all("InaMartini/t5-en-es-translation", df, df)

    print_results(results)
