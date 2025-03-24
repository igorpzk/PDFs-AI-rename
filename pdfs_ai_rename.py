import os
import re
import time
import tiktoken
from PyPDF2 import PdfReader

# If you want to load environment variables from a .env file:
from dotenv import load_dotenv
load_dotenv()

import openai

# Set the OpenAI API key from your .env file
openai.api_key = os.getenv("OPENAI_API_KEY")

max_length = 15000

def get_new_filename_from_openai(pdf_content):
    """
    Uses the OpenAI ChatCompletion endpoint to get a new filename suggestion.
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # or "gpt-3.5-turbo-0125" if you have access
        messages=[
            {
                "role": "system",
                "content": (
                     "You are an assistant tasked with generating filenames for PDF documents. "
                    "Your goal is to create a filename that follows these specific rules:\n"
                    "1. The filename should contain only English alphabet characters (lowercase or uppercase), numbers, and underscores. No spaces, special characters, or punctuation.\n"
                    "2. The filename should not exceed 50 characters in length.\n"
                    "3. The filename should be concise, relevant, and meaningful based on the content of the PDF. If the content is not descriptive enough, you may generate a generic name based on the context.\n"
                    "4. Avoid using words like 'file', 'document', or 'PDF' in the name.\n"
                    "5. Please reply with the filename only (no extra explanations or JSON formatting)."
                ),
            },
            {
                "role": "user",
                "content": pdf_content
            },
        ]
    )

    initial_filename = response["choices"][0]["message"]["content"].strip()
    filename = validate_and_trim_filename(initial_filename)
    return filename

def validate_and_trim_filename(initial_filename):
    """
    Ensures the filename only contains letters, numbers, or underscores.
    Truncates if it's too long or empty. 
    """
    if not initial_filename:
        timestamp = time.strftime('%Y%m%d%H%M%S', time.gmtime())
        return f'empty_file_{timestamp}'
    
    # 1) Allow only alpha-numerics and underscores (any length).
    # 2) Trim if over 50 characters.
    cleaned_filename = re.sub(r'[^A-Za-z0-9_]', '', initial_filename)
    if not cleaned_filename:
        timestamp = time.strftime('%Y%m%d%H%M%S', time.gmtime())
        return f'empty_file_{timestamp}'

    return cleaned_filename[:50]

def rename_pdfs_in_directory(directory):
    """
    Reads all PDFs in a directory, extracts text from the first page, 
    sends it to OpenAI for a suggested new filename, and renames the file.
    """
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    # Sort so the newest files are processed first
    files.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)

    for filename in files:
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(directory, filename)
            print(f"Reading file: {filepath}")

            pdf_content = pdfs_to_text_string(filepath)
            new_file_name = get_new_filename_from_openai(pdf_content)

            # Ensure we don't overwrite an existing PDF
            existing_pdfs = {f for f in os.listdir(directory) if f.lower().endswith(".pdf")}
            if (new_file_name + ".pdf") in existing_pdfs:
                new_file_name += "_01"

            new_filepath = os.path.join(directory, new_file_name + ".pdf")
            try:
                os.rename(filepath, new_filepath)
                print(f"File renamed to {new_filepath}")
            except Exception as e:
                print(f"An error occurred while renaming the file: {e}")

def pdfs_to_text_string(filepath):
    """
    Extracts text from the first page of the PDF file.
    Cuts the text if it exceeds the `max_length` in tokens.
    """
    with open(filepath, 'rb') as file:
        reader = PdfReader(file)
        content = reader.pages[0].extract_text() if reader.pages else ""

        if not content.strip():
            content = "Content is empty or contains only whitespace."
        
        encoding = tiktoken.get_encoding("cl100k_base")
        num_tokens = len(encoding.encode(content))
        if num_tokens > max_length:
            content = content_token_cut(content, num_tokens, max_length)
        return content

def content_token_cut(content, num_tokens, max_length):
    """
    Repeatedly shortens the content until it fits the max token count.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    while num_tokens > max_length:
        # Reduce content length iteratively (e.g., to 90% of current length) 
        # until it fits the allowed token size
        new_length = int(len(content) * 0.9)
        content = content[:new_length]
        num_tokens = len(encoding.encode(content))
    return content

def main():
    directory = ''  # Replace with your PDF directory path or prompt user
    if not directory:
        directory = input("Please input your path:").strip()
    rename_pdfs_in_directory(directory)

if __name__ == "__main__":
    main()
