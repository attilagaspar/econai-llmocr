import json
import openai
import os
import time
import uuid

# Set your OpenAI API key (replace with your actual API key or use an environment variable)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Function to send batch requests to ChatGPT API
def correct_ocr_texts(ocr_entries, model="gpt-4-turbo"):
    """Sends OCR texts to ChatGPT API for correction with unique IDs."""
    prompt = "Please correct the following OCR texts. For each text, return the correction in the format: <OCR_ID>: <Corrected Text>. Do not add any extra comments.\n\n"
    
    for entry in ocr_entries:
        prompt += f"{entry['ocr_id']}: {entry['ocr_text']}\n"
    
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

        # Extract the corrected responses
        corrected_lines = response["choices"][0]["message"]["content"].split("\n")
        corrected_dict = {}

        for line in corrected_lines:
            if ": " in line:
                ocr_id, corrected_text = line.split(": ", 1)
                corrected_dict[ocr_id.strip()] = corrected_text.strip()

        return corrected_dict

    except Exception as e:
        print(f"Error during API call: {e}")
        return {}

# Main function
def process_json(input_file, output_file, batch_size=5):
    """Reads input JSON, corrects OCR texts in batches with unique IDs, and saves output JSON."""
    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Attach unique IDs to each OCR entry
    for entry in data:
        if "ocr_text" in entry and "ocr_id" not in entry:
            entry["ocr_id"] = str(uuid.uuid4())  # Assign a unique ID

    # Process in batches
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        ocr_entries = [{"ocr_id": entry["ocr_id"], "ocr_text": entry["ocr_text"]} for entry in batch if "ocr_text" in entry]

        corrected_dict = correct_ocr_texts(ocr_entries)

        for entry in batch:
            if entry["ocr_id"] in corrected_dict:
                entry["ocr_llm_corrected"] = corrected_dict[entry["ocr_id"]]

        time.sleep(1)  # Avoid hitting API rate limits
    
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

# Example usage
input_json = "ocr_data.json"  # Replace with your actual input JSON file
output_json = "ocr_corrected.json"
batch_size = 3  # Number of texts to send at a time

process_json(input_json, output_json, batch_size)
