import os
import json
import time
import openai

# Define input and output directories
INPUT_FOLDER = "llm_input"
OUTPUT_FOLDER = "ocr_llm_results"
#MODEL = "gpt-4-turbo"  # Adjust model as needed
MODEL = "gpt-4o-mini"
API_KEY = os.environ.get("OPENAI_API_KEY")

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def load_json(filepath):
    """Loads a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(filepath, data):
    """Saves data to a JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def call_chatgpt_api(ocr_text):
    """Sends an OCR text to ChatGPT API and returns the structured response."""
    prompt = (
        "Please correct this OCR-red text which contains information on a Hungarian firm and return a JSON structured as follows. "
        "Please return a field called CORRECTED_TEXT with the whole corrected text. "
        "Please also add as key-value pairs every personal name (as value) and their role (as key) in the firm."
        "Please also add an additional field called PERSONAL_NAMES which contains all personal names extracted from the text. Should be an empty list if there are no personal names."
    )

    retries = 0
    openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Correct client initialization

    while retries < 5:
        try:
            response = openai_client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": ocr_text}
                ]
            )
            return response.choices[0].message.content  # Correct way to access the response

        except openai.APIError as e:  # Correct error handling for OpenAI v1+
            wait_time = 2 ** retries  # Exponential backoff
            print(f"API call failed (attempt {retries + 1}): {e}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            retries += 1

    print("API call failed after multiple attempts. Skipping this element.")
    return None

def process_json_file(input_filepath, output_filepath):
    """Processes a single JSON file, updating its content iteratively."""
    input_data = load_json(input_filepath)
    if os.path.exists(output_filepath):
        output_data = load_json(output_filepath)
    else:
        output_data = []
    
    processed_ids = {item["unique_id"] for item in output_data}  # Assume each element has a unique 'unique_id'
    new_data = output_data.copy()  # Start with the existing data
    number_of_fields = len(input_data)
    current_field = 0
    for item in input_data:
        current_field += 1
        print(f"Processing field {current_field} of {number_of_fields}")
        if item["unique_id"] in processed_ids:
            continue  # Skip already processed items
        
        ocr_text = item.get("ocr_text", "")
        if not ocr_text.strip():
            print(f"Skipping empty OCR text for ID: {item['unique_id']}")
            continue
        
        chatgpt_response = call_chatgpt_api(ocr_text)

        # Delete everything that is not part of the response JSON

        if chatgpt_response:
            # Clean the response to ensure it starts with '{' and ends with '}'
            start_index = chatgpt_response.find("{")
            end_index = chatgpt_response.rfind("}") + 1
            if start_index != -1 and end_index != -1:
                chatgpt_response = chatgpt_response[start_index:end_index]
        print(chatgpt_response)
        if chatgpt_response:
            try:
                item["chatgpt_result"] = json.loads(chatgpt_response)  # Parse response as JSON
            except json.JSONDecodeError:
                print(f"Failed to parse ChatGPT response for ID: {item['unique_id']}")
                with open ("failed_responses.txt", "a") as f:
                    f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Failed to parse ChatGPT response for ID: {item['unique_id']}\n")
                    try:
                        f.write(chatgpt_response)
                    except UnicodeEncodeError:
                        f.write(f"UnicodeEncodeError occurred while writing response to file.\n")
                    f.write("*****************\n\n")
                continue
        item["model_used"] = MODEL
        new_data.append(item)
        save_json(output_filepath, new_data)  # Update output file after each processed element

# Process all JSON files in the input folder
for filename in os.listdir(INPUT_FOLDER):
    if filename.endswith(".json"):
        input_filepath = os.path.join(INPUT_FOLDER, filename)
        output_filepath = os.path.join(OUTPUT_FOLDER, filename)
        print(f"Processing file: {filename}")
        process_json_file(input_filepath, output_filepath)

print("Processing completed.")