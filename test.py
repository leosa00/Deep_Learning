import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

# --- Configuration ---
# NOTE: Update this path if you saved your model to Google Drive!
# Example: MODEL_PATH = "/content/drive/MyDrive/ByT5_Translation_Project/byt5-finetuned"
MODEL_PATH= "malinhauglandh/byt5-en-es-translation"

# Match the prefix and max length used during training
TASK_PREFIX = "translate English to Spanish: "
MAX_LENGTH = 256
# ---------------------

def translate_sentence(model, tokenizer, device, text_to_translate):
    """
    Translates a single English sentence to Spanish using the loaded ByT5 model.
    """
    
    print(f"\n--- Translating ---")
    print(f"EN (Source): {text_to_translate}")

    # 1. Prepare the input text with the mandatory T5 task prefix
    input_text = TASK_PREFIX + text_to_translate

    # 2. Tokenize the input
    input_ids = tokenizer(
        input_text, 
        return_tensors="pt", 
        max_length=MAX_LENGTH, 
        truncation=True
    ).input_ids.to(device)

    # 3. Generate the translation (decoding)
    # We use a context manager to disable gradient calculation for faster inference
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=MAX_LENGTH,
            # Common decoding parameters for quality:
            num_beams=4,
            do_sample=False, # Use beam search for deterministic, high-quality results
            early_stopping=True,
        )

    # 4. Decode the output IDs back to text
    translated_text = tokenizer.decode(output_ids.squeeze(), skip_special_tokens=True)
    
    print(f"ES (Target): {translated_text}")
    print(f"-------------------")
    return translated_text

def main():
    # Set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        # Load the model and tokenizer from the saved path
        print(f"Loading model from: {MODEL_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval() # Set model to evaluation mode

        # --- Test Sentences ---
        test_sentences = [
            "shit",
        
        
        ]
        # ----------------------
        
        for sentence in test_sentences:
            translate_sentence(model, tokenizer, device, sentence)

    except Exception as e:
        print(f"\nFATAL ERROR: Could not load or run the model.")
        print(f"Please check your MODEL_PATH variable: {MODEL_PATH}")
        print(f"Error details: {e}")

if __name__ == "__main__":
    main()