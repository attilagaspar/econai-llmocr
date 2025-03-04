
import json
import openai
import os
import time

# Set your OpenAI API key (replace with your actual API key or use an environment variable)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Function to send batch requests to ChatGPT API
def correct_ocr_texts(ocr_texts, model="gpt-4-turbo"):
    """Sends OCR texts to ChatGPT API for correction."""
    prompt = "Please correct the following OCR texts. Please refrain from adding any comments, such as 'here are the corrected versions of the OCR texts' or comments on how well it went:\n\n"
    for i, text in enumerate(ocr_texts):
        prompt += f"{i+1}. {text}\n"
    
    try:
        response = openai.ChatCompletion.create(
            model=model,
            api_key=OPENAI_API_KEY,  # Using the API key properly
            messages=[
                {"role": "system", "content": "You are an expert in OCR text correction."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        corrected_texts = response["choices"][0]["message"]["content"].split("\n")
        print(corrected_texts)
        return [text.split(". ", 1)[1] if ". " in text else text for text in corrected_texts]
    except Exception as e:
        print(f"Error during API call: {e}")
        return ["ERROR"] * len(ocr_texts)  # Return error placeholders

# Main function
def process_json(input_file, output_file, batch_size=5):
    """Reads input JSON, corrects OCR texts in batches, and saves output JSON."""
    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        ocr_texts = [entry["ocr_text"] for entry in batch if "ocr_text" in entry]

        corrected_texts = correct_ocr_texts(ocr_texts)

        for j, entry in enumerate(batch):
            entry["ocr_llm_corrected"] = corrected_texts[j]

        time.sleep(1)  # Avoid hitting API rate limits
    
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

# Example usage
input_json = "ocr_data.json"  # Replace with your actual input JSON file
output_json = "ocr_corrected.json"
batch_size = 3  # Number of texts to send at a time

process_json(input_json, output_json, batch_size)
