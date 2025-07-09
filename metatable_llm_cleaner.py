import os
import json
import openai
import base64
import io
from PIL import Image

#This script opens the meta-tables from the /metatable_raw folder, 
#Sends the OCR texts and the image segment to the OpenAI API for correction 
#Then saves the corrected texts in a new JSON now located in 
# the /metatable_llm folder.


# Define paths
raw_folder = "metatable_raw"
output_folder = "metatable_llm"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Function to send text to OpenAI API for correction
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),  # Ensure the API key is set in the environment
)

def correct_text_with_openai(text: str, image: Image.Image) -> str:
    """
    Compare OCR text to the image content and return corrected text using GPT-4o.

    Args:
        text (str): The OCR text to be corrected.
        image (PIL.Image.Image): PIL Image object containing the image to compare.

    Returns:
        str: Corrected text output from the model.
    """
    # Convert PIL Image object to bytes in PNG format
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    b64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Prepare instructions and input for the OpenAI API
    
    """
    # Instructions for the OpenAI API when OCR text is provided
    instructions = (
        "You are a coding assistant. Compare the OCR text to the image content "
        "and correct any errors. Provide only the corrected text in your response."
    )
    input_text = (
        f"Here is some OCR text extracted from this image:\n{text}\n\n"
        "Below is the base64-encoded image content:\n"
        f"{b64_image}\n\n"
        "Please carefully compare the text to the image content and correct any errors. Only return the corrected text.\n"
        "Please also follow these instructions:\n"
        "Check if the length of the text in the image is similar to the length of the text in the OCR. Sometimes the decimal point is mistaken for a zero.\n"
        "The first digit of a number is never a zero, so if you see such, that is an OCR error.\n"
        "If you see a single dash or hyphen or horizontal line in the image in a row, please replace it with a zero in the text.\n"
        "It is very important to not disregard any of these (there might be more than one of these one after the other) as this will shift every following rows upwards. \n"
        "Please double-check that the number of rows in the image is the same as the number of rows in the text.\n"
        "If you see a sequence of dots or other repeating characters after a string of non-numeric characters, please just remove it.\n"
        "If you see digits in such a format DD-DD in the image (digits 'minus sign' digits), the - is a decimal point. Please correct the text accordingly\n"
    )
    """
    # Instructions for the OpenAI API when OCR text is not provided
    instructions = (
        "Can you read this for me?"
        "Your answer should be plain text, without any additional formatting or explanations. New lines in the image should be represented in your response with newline characters."
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an expert at reading numbers from images and correcting OCR errors."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": f"""Can you read this b64 image for me? Your answer should be plain text, without any additional formatting or explanations. New lines in the image should be represented in your response with newline characters. Lone dashes in a table row are corresponding to zeros (more than one can follow one another); decimal points are usually represented with a dot but they are placed higher relative to the digit than usual. Please always make sure that the number of rows in the image is the same as the number of rows in the text. If you see a sequence of dots or other repeating characters after a string of non-numeric characters, please just remove it. If you see digits in such a format DD-DD in the image (digits 'minus sign' digits), the - is a decimal point. Please correct the text accordingly.\n\n"
                                You can use this tesseract output as help, but be aware that it is full of errors:\n{text}\n\n"""

                    }
                ]
            }
        ]
    )

    corrected_text = response.choices[0].message.content.strip()
    print(f"Corrected text: {corrected_text}")
    return corrected_text
# Process each file in the raw folder
for filename in os.listdir(raw_folder):
    if filename.endswith(".json"):
        print(f"Processing file: {filename}")
        # Construct full file paths
        raw_file_path = os.path.join(raw_folder, filename)
        output_file_path = os.path.join(output_folder, filename)

        # Load input JSON
        with open(raw_file_path, "r", encoding="utf-8") as raw_file:
            try:
                data = json.load(raw_file)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in file {filename}: {e}")
                continue

        # Check if output JSON exists
        if os.path.exists(output_file_path):
            with open(output_file_path, "r", encoding="utf-8") as output_file:
                try:
                    output_data = json.load(output_file)
                except json.JSONDecodeError as e:
                    print(f"Error decoding existing output JSON in file {filename}: {e}")
                    output_data = {"shapes": []}
        else:
            output_data = {"shapes": []}

        # Create a set of already corrected entries
        corrected_entries = {entry["id"] for entry in output_data["shapes"] if "corrected_text" in entry}

        # Open the corresponding image
        try:
            metaimage = Image.open(os.path.join(raw_folder, filename.replace(".json", ".jpg")))
        except FileNotFoundError:
            print(f"Image file not found for {filename}. Skipping...")
            continue
        
        # Nunmber of pre-processed entries
        processed_entries = len(output_data["shapes"])
        # Correct texts in the JSON
        for entry in data["shapes"]:
            total_entries = len(data["shapes"])
            
            current_entry = data["shapes"].index(entry) + 1

            if "ocr_text" in entry and entry.get("id") not in corrected_entries:
                print(f"Processing entry {current_entry} of {total_entries-processed_entries}.")
                print(f"Column {entry['column']} row {entry['row']}.")
                # Extract bounding box from "points" field
                if "points" in entry:
                    points = entry["points"]
                    try:
                        # Ensure points are valid and ordered correctly
                        x1, y1 = map(int, points[0])  # First point
                        x2, y2 = map(int, points[1])  # Second point
                        x1, x2 = sorted([x1, x2])  # Ensure x1 <= x2
                        y1, y2 = sorted([y1, y2])  # Ensure y1 <= y2

                        # Crop the image
                        local_img = metaimage.crop((x1, y1, x2, y2))
                        entry["corrected_text"] = correct_text_with_openai(entry["ocr_text"], local_img)

                        # Add the corrected entry to the output data
                        output_data["shapes"].append(entry)

                        # Save the output JSON after every API call
                        with open(output_file_path, "w", encoding="utf-8") as output_file:
                            json.dump(output_data, output_file, indent=4)
                    except Exception as e:
                        print(f"Error processing bounding box for {filename}: {e}")
                        entry["corrected_text"] = entry["ocr_text"]  # Fallback to original text
                        output_data["shapes"].append(entry)
                        with open(output_file_path, "w", encoding="utf-8") as output_file:
                            json.dump(output_data, output_file, indent=4)
                else:
                    print(f"No 'points' field in entry for {filename}. Skipping correction.")
                    entry["corrected_text"] = entry["ocr_text"]  # Fallback to original text
                    output_data["shapes"].append(entry)
                    with open(output_file_path, "w", encoding="utf-8") as output_file:
                        json.dump(output_data, output_file, indent=4)

        print(f"Processed and saved: {filename}")