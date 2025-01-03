'''This file is used to create "train.json" & "test.json" file. 
The program will tell you to select files, the files are selected from Hybrid-DeepCom Dataset.'''

import os
import sys
import json
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm  # For the progress bar

# Add the parent directory (ALSI_Transformer) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from get_CAT import generate_code_aligned_type_sequence

# Initialize Tkinter and hide the root window.
root = tk.Tk()
root.withdraw()

# Select the code file.
code_file_path = filedialog.askopenfilename(title="Select the file containing code snippets")
if not code_file_path:
    print("No file selected for code snippets.")
    exit()

# Select the comment file.
comment_file_path = filedialog.askopenfilename(title="Select the file containing comments")
if not comment_file_path:
    print("No file selected for comments.")
    exit()

# Prompt the user to enter the output file name and location.
output_file_path = filedialog.asksaveasfilename(
    title="Save the output JSON file",
    defaultextension=".json",
    filetypes=[("JSON files", "*.json")],
    initialfile="output.json"
)
if not output_file_path:
    print("No output file name provided.")
    exit()

# Initialize the list to store data.
data = []

# Open both files and read line by line.
with open(code_file_path, 'r') as code_file, open(comment_file_path, 'r') as comment_file:
    # Use tqdm to show the progress bar for the loop.
    for code_line, comment_line in tqdm(zip(code_file, comment_file), total=sum(1 for _ in open(code_file_path)), desc="Processing"):
        # Clean up lines.
        code_line = code_line.strip()
        comment_line = comment_line.strip()

        # Create a dictionary for each line.
        entry = {
            "code": code_line,
            "CAT": generate_code_aligned_type_sequence(code_line),
            "comment": comment_line
        }

        # Add entry to data list.
        data.append(entry)

# Save the list of dictionaries as a JSON file.
with open(output_file_path, 'w') as json_file:
    json.dump(data, json_file, indent=4)

print(f"JSON file created successfully at {output_file_path}")
