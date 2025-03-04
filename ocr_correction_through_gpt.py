import json
import openai
import os
import time
import uuid

# Set your OpenAI API key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Function to send batch requests to ChatGPT API
def correct_ocr_texts(ocr_entries, model="gpt-4-turbo"):
    """Sends OCR texts to ChatGPT API for correction and name extraction with unique IDs."""
    
    prompt = (
        "Please correct the following OCR texts and extract any personal names.\n"
        "Return the data in the format:\n"
        "<OCR_ID>: <Corrected Text>\n"
        "<OCR_ID>_names: Name1, Name2, ...\n"
        "If there are no names, return an empty list.\n\n"
    )

    for entry in ocr_entries:
        prompt += f"{entry['ocr_id']}: {entry['ocr_text']}\n"

    try:
        response = openai.ChatCompletion.create(
            model=model,
            api_key=OPENAI_API_KEY,
            messages=[
                {"role": "system", "content": "You are an expert in OCR text correction and name extraction."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        # Ensure we got a valid response
        if not response or "choices" not in response or not response["choices"]:
            print("⚠️ API returned an empty response.")
            return {}, {}

        response_text = response["choices"][0]["message"]["content"]
        corrected_dict = {}
        names_dict = {}

        # Process each line of response
        for line in response_text.split("\n"):
            if ": " in line:
                key, value = line.split(": ", 1)
                key = key.strip()
                value = value.strip()

                if key.endswith("_names"):  # Extracting names
                    names = [name.strip() for name in value.split(",") if name.strip()]
                    names_dict[key.replace("_names", "")] = names
                else:  # Extracting corrected text
                    corrected_dict[key] = value

        return corrected_dict, names_dict

    except Exception as e:
        print(f"🚨 Error during API call: {e}")
        return {}, {}

# Main function
def process_json(input_file, output_file, batch_size=5):
    """Reads input JSON, corrects OCR texts in batches, extracts names, and saves output JSON."""
    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Attach unique IDs to each OCR entry if missing
    for entry in data:
        if "ocr_text" in entry and "ocr_id" not in entry:
            entry["ocr_id"] = str(uuid.uuid4())  # Assign a unique ID

    # Process in batches
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        ocr_entries = [{"ocr_id": entry["ocr_id"], "ocr_text": entry["ocr_text"]} for entry in batch if "ocr_text" in entry]

        corrected_dict, names_dict = correct_ocr_texts(ocr_entries)

        for entry in batch:
            if entry["ocr_id"] in corrected_dict:
                entry["ocr_llm_corrected"] = corrected_dict[entry["ocr_id"]]
                entry["extracted_names"] = names_dict.get(entry["ocr_id"], [])

        time.sleep(1)  # Avoid hitting API rate limits
    
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

# Example usage
input_json = "ocr_data.json"  # Replace with your actual input JSON file
output_json = "ocr_corrected_with_names.json"
batch_size = 3  # Number of texts to send at a time

process_json(input_json, output_json, batch_size)
