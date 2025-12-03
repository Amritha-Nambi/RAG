import fitz
import os
import re

# Use the ISBN as a constant delimiter
ISBN_DELIMITER = "ISBN 978-1-78110-647-1"

def clean_text(text: str) -> str:
    """
    Cleans up common non-unicode characters and formatting issues.
    """
    # Replace common curly quotes and apostrophes with standard ones
    text = text.replace("’", "'")
    text = text.replace("‘", "'")
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("„", '"')
    text = text.replace("—", "-")
    text = text.replace("–", "-")
    text = text.replace("…", "...")
    
    # Remove extra spaces around punctuation
    text = re.sub(r'\s+([?.!,"])', r'\1', text)
    
    # Normalize multiple newlines to a consistent number (e.g., two for a new paragraph)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove any stray spaces at the beginning of lines
    text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)
    
    return text

def extract_text_from_pdfs(pdf_folder_path: str, output_folder_path: str, book_titles: list):
    """
    Extracts text from a single PDF using block-based extraction for better paragraph preservation.
    """
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
        print(f"Created output directory: {output_folder_path}")

    if not os.path.exists(pdf_folder_path):
        print(f"Error: PDF input folder not found at '{pdf_folder_path}'. Please create it and add your PDF file.")
        return

    pdf_file_name = next((f for f in os.listdir(pdf_folder_path) if f.endswith(".pdf")), None)
    if not pdf_file_name:
        print("Error: No PDF file found in the specified folder.")
        return

    pdf_path = os.path.join(pdf_folder_path, pdf_file_name)
    print(f"Processing the single PDF: {pdf_file_name}")

    full_text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            # Extract text blocks (paragraphs)
            blocks = page.get_text("blocks")
            blocks.sort(key=lambda block: (block[1], block[0]))  # Sort by vertical then horizontal position
            
            for block in blocks:
                block_text = block[4].strip()  # The actual text is at index 4
                if block_text:  # Only add non-empty blocks
                    # Check if this block is likely a paragraph continuation
                    if not full_text.endswith('\n\n') and full_text and not full_text[-1] in '.!?"\n':
                        full_text += " " + block_text + "\n"
                    else:
                        full_text += block_text + "\n"
            
            full_text += "\n"  # Add separation between pages
        
        doc.close()
    except Exception as e:
        print(f"Could not read the PDF file. Error: {e}")
        return

    # Clean the text
    full_text = clean_text(full_text)

    # Split the entire text into books using the ISBN as the delimiter
    books_text = full_text.split(ISBN_DELIMITER)
    
    books_text = [book for book in books_text if len(book.strip()) > 50]

    if len(books_text) == 0:
        print("Error: Could not find the ISBN delimiter to split the books.")
        return

    print(f"Found and extracted {len(books_text)} books.")

    # Save each book's text to a new file
    for i, book_content in enumerate(books_text):
        if i < len(book_titles):
            title = book_titles[i]
        else:
            title = f"book_{i+1}"
        
        # Format the title for a valid filename (lowercase and underscores)
        output_txt_file_name = f"{title.lower().replace(' ', '_')}.txt"
        output_txt_path = os.path.join(output_folder_path, output_txt_file_name)
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(book_content.strip())
        print(f"Saved: {output_txt_file_name}")

    print("\nExtraction and splitting complete!")
