import os
import sys
import json
import base64
import io
from PIL import Image
from openai import OpenAI

USE_OCR_TEXT_FOR_PROMPT = False  # Set to False to always use the else branch


def find_labelme_jsons(input_dir):
    json_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files

def load_image_for_json(json_path):
    base = os.path.splitext(json_path)[0]
    for ext in ['.jpg', '.jpeg', '.png']:
        img_path = base + ext
        if os.path.exists(img_path):
            return img_path
    return None

def encode_image_b64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def call_openai_api(image, prompt, model="gpt-4o-mini"):
    b64_image = encode_image_b64(image)
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
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
                        "text": prompt
                    }
                ]
            }
        ]
    )
    return response.choices[0].message.content.strip(), model

def main():
    if len(sys.argv) < 2:
        print("Usage: python metatable_llm_cleaner.py <input_dir>")
        sys.exit(1)
    input_dir = sys.argv[1]

    json_files = find_labelme_jsons(input_dir)
    print(f"Found {len(json_files)} LabelMe JSON files.")

    for json_path in json_files:
        img_path = load_image_for_json(json_path)
        if not img_path:
            print(f"No image found for {json_path}, skipping.")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        img = Image.open(img_path)
        print(f"Processing number of shapes in {json_path}: {len(data.get('shapes', []))}")
        for shape in data.get("shapes", []):
            # Only process if label is "text_cell" or "column_header"
            if shape.get("label") not in ("text_cell", "column_header"):
                continue
            # Skip if already processed
            if "openai_output" in shape:
                print(f"Skipping already processed shape in {json_path}")
                continue
        

            points = shape.get("points", [])
            if len(points) < 2:
                continue
            try:
                x1, y1 = map(int, points[0])
                x2, y2 = map(int, points[1])
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
                snippet = img.crop((x1, y1, x2, y2))
            except Exception as e:
                print(f"Error cropping image for {json_path}: {e}")
                continue

            # Choose prompt based on tesseract_output
            if USE_OCR_TEXT_FOR_PROMPT and "tesseract_output" in shape and "ocr_text" in shape["tesseract_output"]:            
            # if "tesseract_output" in shape and "ocr_text" in shape["tesseract_output"]:
                ocr_text = shape["tesseract_output"]["ocr_text"]
                prompt = (
                    "Here is some OCR text extracted from this image:\n"
                    f"{ocr_text}\n\n"
                    "Please carefully compare the text to the image content and correct any errors. Only return the corrected text, no accompanying text like 'here is the corrected text' etc.\n"
                    "Please also follow these instructions:\n"
                    "Check if the length of the text in the image is similar to the length of the text in the OCR. Sometimes the decimal point is mistaken for a zero.\n"
                    "The first digit of a number is never a zero, so if you see such, that is an OCR error.\n"
                    "Lone dashes (hyphens, underscores) in a table row are important because they represent missing data, please don't remove them \n"
                    "It is very important to not disregard any of these (there might be more than one of these one after the other) as this will shift every following rows upwards. \n"
                    "Please double-check that the number of rows in the image is the same as the number of rows in the text.\n"
                    "If you see a sequence of dots or other repeating characters after a string of non-numeric characters, please just remove it.\n"
                    "If you see digits in such a format DD-DD in the image (digits 'minus sign' digits), the - is a decimal point. Please correct the text accordingly\n"
                )
            else:
                #prompt = (
                #    "Can you read this b64 image for me? Your answer should be plain text, without any additional formatting or explanations. Only return the corrected text, no accompanying text like 'here is the corrected text'"
                #    "New lines in the image should be represented in your response with newline characters. "
                #    "If you see a sequence of dots or other repeating characters after a string of non-numeric characters, please just remove it. "
                #)
                prompt = (
                    "Read the following base64 image. Return only the corrected plain text, using newline characters for line breaks. Dashes are frequently used to represent missing data in tables, so do not remove them. "
                    "Do not include any explanations or formatting. If a non-numeric string is followed by a sequence of dots or repeating characters, remove those characters."
                )
                #                     "Lone dashes (hyphens, underscores) in a table row are important because they represent missing data, please don't remove them (more than one can follow one another); decimal points are usually represented with a dot but they are placed higher relative to the digit than usual. "
                #    "Please always make sure that the number of rows in the image is the same as the number of rows in the text. "
                #                     "If you see digits in such a format DD-DD in the image (digits 'minus sign' digits), the - is a decimal point. Please correct the text accordingly."

            try:
                result, model = call_openai_api(snippet, prompt)
                shape["openai_output"] = {
                    "response": result,
                    "model": model
                }
                print(f"Processed shape in {json_path}")
                print(f"OpenAI response: {result}")
                # Save the updated JSON after each successful API call
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"OpenAI API error for {json_path}: {e}")
                continue

        # Save the updated JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved: {json_path}")

if __name__ == "__main__":
    main()