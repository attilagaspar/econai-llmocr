import json
import openai
import os
import time
import uuid
import sys
import glob

# Set your OpenAI API key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# model
mdl = "gpt-4-turbo"

# Define input and output folders
INPUT_FOLDER = "ocr_results"
OUTPUT_FOLDER = "ocr_llm_results"

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Function to send batch requests to ChatGPT API
def correct_ocr_texts(ocr_entries, model=mdl):
#def correct_ocr_texts(ocr_entries, model="gpt-3.5-turbo"):
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

    print(f"🔄 Sending batch of {len(ocr_entries)} texts to OpenAI API...")  # Debug log before API call

    try:
        response = openai.ChatCompletion.create(
            model=model,
            api_key=OPENAI_API_KEY,
            messages=[
                {"role": "system", "content": "You are an expert in OCR text correction and name extraction."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            request_timeout=30  # ⏳ Added timeout to avoid hanging requests
        )

        print("✅ API call successful.")  # Debug log after API call

        # Ensure valid response
        if not response or "choices" not in response or not response["choices"]:
            print("⚠️ API returned an empty response.")
            return {}, {}

        response_text = response["choices"][0]["message"]["content"]
        corrected_dict = {}
        names_dict = {}

        # Process response line by line
        for line in response_text.split("\n"):
            if ": " in line:
                key, value = line.split(": ", 1)
                key = key.strip()
                value = value.strip()

                if key.endswith("_names"):
                    names = [name.strip() for name in value.split(",") if name.strip()]
                    names_dict[key.replace("_names", "")] = names
                else:
                    corrected_dict[key] = value

        return corrected_dict, names_dict

    except Exception as e:
        print(f"🚨 Error during API call: {e}")
        return {}, {}

# Main function
def process_json(input_file, output_file, batch_size=3):
    """Reads input JSON, corrects OCR texts in batches, extracts names, and saves output JSON."""
    print(f"📄 Processing file: {input_file}")  # ✅ Now prints filename immediately

    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Attach unique IDs to each OCR entry if missing
    for entry in data:
        if "ocr_text" in entry and "ocr_id" not in entry:
            entry["ocr_id"] = str(uuid.uuid4())  # Assign a unique ID

    # Calculate total number of batches
    total_batches = (len(data) + batch_size - 1) // batch_size  # Round up division

    # Process in batches
    for batch_num, i in enumerate(range(0, len(data), batch_size), start=1):
        batch = data[i:i+batch_size]
        ocr_entries = [{"ocr_id": entry["ocr_id"], "ocr_text": entry["ocr_text"]} for entry in batch if "ocr_text" in entry]

        print(f"🔄 Processing batch {batch_num}/{total_batches}...")  # ✅ Now prints batch status immediately

        corrected_dict, names_dict = correct_ocr_texts(ocr_entries)

        for entry in batch:
            if entry["ocr_id"] in corrected_dict:
                entry["ocr_llm_corrected"] = corrected_dict[entry["ocr_id"]]
                entry["extracted_names"] = names_dict.get(entry["ocr_id"], [])

        time.sleep(1)  # Avoid hitting API rate limits
    
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    print(f"✅ Finished processing: {output_file}")  # ✅ Now prints when done

# Function to process all `.bbocr` files OR a single file
def main():
    """Process either all .bbocr files or a single file based on input argument."""
    
    if len(sys.argv) > 1:
        # User provided a specific file
        input_filename = sys.argv[1]
        input_path = os.path.join(INPUT_FOLDER, input_filename)
        if not os.path.exists(input_path):
            print(f"❌ Error: File '{input_filename}' not found in {INPUT_FOLDER}")
            sys.exit(1)
        
        output_filename = input_filename.replace(".bbocr", f"_{mdl}_corrected.json")
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        print(f"📄 Processing single file: {input_filename}")
        process_json(input_path, output_path)

    else:
        # Process all .bbocr files
        bbocr_files = glob.glob(os.path.join(INPUT_FOLDER, "*.bbocr"))

        if not bbocr_files:
            print(f"⚠️ No .bbocr files found in {INPUT_FOLDER}")
            sys.exit(1)

        for input_path in bbocr_files:
            filename = os.path.basename(input_path)
            output_filename = filename.replace(".bbocr", f"_{mdl}_corrected.json")
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            
            print(f"📄 Processing file: {filename}")
            process_json(input_path, output_path)

# Run the script
if __name__ == "__main__":
    main()
