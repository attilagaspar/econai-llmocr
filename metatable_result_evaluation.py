# This script is designed to review JSON files containing OCR and LLM results.
# It allows users to visually inspect the results, make corrections, and save their choices.
# The script uses a GUI to display images and corresponding OCR and LLM texts, allowing users to select the correct option for each entry.
# The results are saved in a JSON file for later analysis.


import os
import json
import random
from tkinter import Tk, Label, Button, Radiobutton, StringVar, Frame, filedialog, Entry
from PIL import Image, ImageTk
from datetime import datetime  # Add this import at the top of your file

# Load JSON and corresponding image
def load_json_and_image(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    image_path = json_path.replace(".json", ".jpg")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    image = Image.open(image_path)
    return data, image

# Crop the image based on points
def crop_image(image, points):
    x1, y1 = map(int, points[0])
    x2, y2 = map(int, points[1])
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    return image.crop((x1, y1, x2, y2))

# Main application class
class JSONReviewerApp:
    def __init__(self, root, json_path):
        self.root = root
        self.root.title("JSON Reviewer")
        self.json_path = json_path
        self.data, self.image = load_json_and_image(json_path)
        self.current_entry = None
        self.results = []
        self.reviewed_entries = self.load_reviewed_entries()  # Load reviewed entries
        self.unreviewed_entries = self.get_unreviewed_entries()  # Filter unreviewed entries
        self.total_entries = self.calculate_total_entries()  # Total rows across unreviewed entries

        # Store the application start time
        self.start_time = datetime.now().isoformat()  # Store the start time in ISO format



        # Statistics tracking
        self.stats = {"ocr": 0, "llm": 0, "both": 0, "neither": 0}

        # Create UI elements
        self.stats_label = Label(root, text="", font=("Arial", 12))
        self.stats_label.pack(pady=10)

        self.main_frame = Frame(root)
        self.main_frame.pack()

        # Image display
        self.image_label = Label(self.main_frame)
        self.image_label.grid(row=0, column=0, rowspan=2, padx=10, pady=10)

        # Table for OCR and LLM outputs
        self.table_frame = Frame(self.main_frame)
        self.table_frame.grid(row=0, column=1, padx=10, pady=10)

        self.radio_buttons = []  # Store radio button variables for each row
        self.text_entries = []  # Store text entry fields for corrections

        self.submit_button = Button(root, text="Submit", command=self.submit_choice)
        self.submit_button.pack(pady=10)

        # Bind keys for quick selection
        #self.root.bind("o", lambda event: self.quick_submit("ocr"))
        #self.root.bind("l", lambda event: self.quick_submit("llm"))
        #self.root.bind("b", lambda event: self.quick_submit("both"))
        #self.root.bind("n", lambda event: self.quick_submit("neither"))
        self.root.bind("<Return>", lambda event: self.submit_choice())
        self.load_random_entry()

    def calculate_total_entries(self):
        """Calculate the total number of rows (max of OCR and LLM rows) across all unreviewed entries."""
        total = 0
        for entry in self.unreviewed_entries:  # Only consider unreviewed entries
            ocr_text = entry.get("ocr_text", "")
            llm_text = entry.get("corrected_text", "")
            ocr_lines = ocr_text.split("\n")
            llm_lines = llm_text.split("\n")
            total += max(len(ocr_lines), len(llm_lines))  # Add the maximum row count for this entry
        return total

    def load_reviewed_entries(self):
        """Load reviewed entries from review_results.json."""
        if os.path.exists("review_results.json"):
            with open("review_results.json", "r", encoding="utf-8") as f:
                reviewed_data = json.load(f)
                self.results = reviewed_data  # Initialize self.results with existing results
                return {(entry["row"], entry["column"]) for entry in reviewed_data}
        return set()

    def get_unreviewed_entries(self):
        """Filter out entries that have already been reviewed."""
        unreviewed = []
        for entry in self.data["shapes"]:
            row = entry["row"]
            column = entry["column"]
            if (row, column) not in self.reviewed_entries:
                unreviewed.append(entry)
        return unreviewed

    def load_random_entry(self):
        """Load a random unreviewed entry."""
        if not self.unreviewed_entries:
            self.stats_label.config(text="All entries have been reviewed!")
            return

        self.current_entry = random.choice(self.unreviewed_entries)
        self.unreviewed_entries.remove(self.current_entry)  # Remove from unreviewed list
        points = self.current_entry["points"]
        cropped_image = crop_image(self.image, points)

        # Resize the image to 50% of its original size
        resized_image = cropped_image.resize(
            (cropped_image.width // 2, cropped_image.height // 2)
        )
        self.tk_image = ImageTk.PhotoImage(resized_image)
        self.image_label.config(image=self.tk_image)

        # Clear the table frame
        for widget in self.table_frame.winfo_children():
            widget.destroy()

        # Display OCR and LLM texts as a table
        ocr_text = self.current_entry.get("ocr_text", "")
        llm_text = self.current_entry.get("corrected_text", "")
        ocr_lines = ocr_text.split("\n")
        llm_lines = llm_text.split("\n")
        max_lines = max(len(ocr_lines), len(llm_lines))

        self.radio_buttons = []  # Reset radio button variables
        self.text_entries = []  # Reset text entry fields

        for i in range(max_lines):
            ocr_line = ocr_lines[i] if i < len(ocr_lines) else ""
            llm_line = llm_lines[i] if i < len(llm_lines) else ""

            # Determine if the lines are different
            is_different = ocr_line.strip() != llm_line.strip()

            # Add OCR text to the table
            Label(
                self.table_frame,
                text=ocr_line,
                font=("Arial", 10, "bold") if is_different else ("Arial", 12),
                fg="red" if is_different else "black",
                anchor="w",
                width=10  # Decrease width to reduce horizontal distance
            ).grid(row=i * 2, column=0, sticky="w")  # Use row=i*2 to leave space for the separator

            # Add LLM text to the table
            Label(
                self.table_frame,
                text=llm_line,
                font=("Arial", 10, "bold") if is_different else ("Arial", 12),
                fg="red" if is_different else "black",
                anchor="w",
                width=10  # Decrease width to reduce horizontal distance
            ).grid(row=i * 2, column=1, sticky="w")

            # Add radio buttons for each row
            choice_var = StringVar(value="both")
            self.radio_buttons.append(choice_var)
            Radiobutton(self.table_frame, text="OCR", variable=choice_var, value="ocr").grid(row=i * 2, column=2, sticky="w")
            Radiobutton(self.table_frame, text="LLM", variable=choice_var, value="llm").grid(row=i * 2, column=3, sticky="w")
            Radiobutton(self.table_frame, text="Both", variable=choice_var, value="both").grid(row=i * 2, column=4, sticky="w")
            Radiobutton(self.table_frame, text="Neither", variable=choice_var, value="neither").grid(row=i * 2, column=5, sticky="w")

            # Add text entry field for corrections
            text_entry = Entry(self.table_frame, width=30)
            text_entry.grid(row=i * 2, column=6, sticky="w")
            self.text_entries.append(text_entry)

            # Add a horizontal black line to separate rows
            Frame(self.table_frame, height=1, width=800, bg="black").grid(row=i * 2 + 1, column=0, columnspan=7, pady=5)

    def submit_choice(self):
        """Submit the user's choices and save them."""
        result = {
            "row": self.current_entry["row"],
            "column": self.current_entry["column"],
            "choices": [],
            "corrections": [],
            "timestamp": datetime.now().isoformat(),  # Add the current timestamp
             "start_time": self.start_time  # Add the application start time
        }
        for choice_var, text_entry in zip(self.radio_buttons, self.text_entries):
            result["choices"].append(choice_var.get())
            result["corrections"].append(text_entry.get() if choice_var.get() == "neither" else "")

        # Append the result to the in-memory results list
        self.results.append(result)

        # Update statistics
        for choice in result["choices"]:
            self.stats[choice] += 1

        # Increment the reviewed entries count
        self.reviewed_entries.add((self.current_entry["row"], self.current_entry["column"]))

        # Update the stats label
        self.update_stats_label()

        # Load the next random entry
        self.load_random_entry()

        # Save results to the output file
        self.save_results()

    def save_results(self):
        """Save the results to review_results.json."""
        with open("review_results.json", "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=4)

    def quick_submit(self, choice):
        """Quickly select a choice for all rows and submit."""
        for choice_var in self.radio_buttons:
            choice_var.set(choice)  # Set the choice for all rows
        self.submit_choice()
        
    def update_stats_label(self):
        """Update the statistics label with the current counts and percentages."""
        total_reviewed = sum(self.stats.values())
        if total_reviewed == 0:
            stats_text = f"0 entries reviewed so far out of {self.total_entries}."
        else:
            ocr_percent = (self.stats["ocr"] / total_reviewed) * 100
            llm_percent = (self.stats["llm"] / total_reviewed) * 100
            both_percent = (self.stats["both"] / total_reviewed) * 100
            neither_percent = (self.stats["neither"] / total_reviewed) * 100
            stats_text = (
                f"{total_reviewed} entries reviewed so far out of {self.total_entries}. "
                f"{ocr_percent:.2f}% OCR, {llm_percent:.2f}% LLM, "
                f"{both_percent:.2f}% BOTH, {neither_percent:.2f}% NEITHER"
            )
        self.stats_label.config(text=stats_text)

# Main function
def main():
    # Ask the user to select a JSON file
    json_path = filedialog.askopenfilename(title="Select JSON File", filetypes=[("JSON Files", "*.json")])
    if not json_path:
        print("No file selected. Exiting.")
        return

    # Create the application window
    root = Tk()
    app = JSONReviewerApp(root, json_path)
    root.mainloop()

if __name__ == "__main__":
    main()