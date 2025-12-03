from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import re


def load_documents(folder_path: str) -> list[Document]:
    """
    Loads all text files from a folder into a list of LangChain Documents.
    """
    print(f"Loading documents from: {folder_path}")
    
    # Check if the directory exists
    if not os.path.isdir(folder_path):
        print(f"Error: Directory not found at '{folder_path}'")
        return []
        
    documents = []
    # Loop through all files in the specified folder
    for filename in os.listdir(folder_path):
        # Only process files that end with .txt
        if filename.endswith(".txt"):
            # Create the full path to the file
            filepath = os.path.join(folder_path, filename)
            
            try:
                # Open and read the file with UTF-8 encoding
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Create a metadata dictionary
                # The source filename is useful for tracking where the text came from
                metadata = {"source": filename}
                
                # Create a Document object and add it to our list
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)

            except Exception as e:
                print(f"Could not read or process file '{filename}'. Error: {e}")

    print(f"Successfully loaded {len(documents)} documents.")
    return documents


def extract_chapter_number(text: str) -> str:
    """
    Extracts the chapter number from text that contains a chapter heading.
    
    Args:
        text: Text that may contain a chapter heading like "CHAPTER ONE:" or "CHAPTER 1:"
        
    Returns:
        Chapter number as a string (e.g., "ONE", "1", "TWENTY-THREE") or "Unknown"
    """
    # Pattern to match chapter headings
    # Matches: "CHAPTER ONE:", "CHAPTER 1:", "CHAPTER TWENTY-THREE:", etc.
    pattern = r'CHAPTER\s+(ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|' \
              r'ELEVEN|TWELVE|THIRTEEN|FOURTEEN|FIFTEEN|SIXTEEN|SEVENTEEN|' \
              r'EIGHTEEN|NINETEEN|TWENTY|TWENTY-ONE|TWENTY-TWO|TWENTY-THREE|' \
              r'TWENTY-FOUR|TWENTY-FIVE|TWENTY-SIX|TWENTY-SEVEN|TWENTY-EIGHT|' \
              r'TWENTY-NINE|THIRTY|THIRTY-ONE|THIRTY-TWO|THIRTY-THREE|' \
              r'THIRTY-FOUR|THIRTY-FIVE|THIRTY-SIX|THIRTY-SEVEN|\d+)'
    
    match = re.search(pattern, text, re.IGNORECASE)
    
    if match:
        return match.group(1)  # Returns the chapter number (e.g., "ONE", "23")
    
    return "Unknown"


def split_documents_into_chunks(documents: list[Document]) -> list[Document]:
    """
    Splits a list of documents into smaller, semantically meaningful chunks.
    Each chunk includes metadata about which chapter it came from.
    """
    print("Splitting documents into chunks...")
    
    all_chunks = []
    
    for doc in documents:
        # Split each document by chapters first
        chapter_splits = re.split(
            r'(CHAPTER\s+(?:ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|'
            r'ELEVEN|TWELVE|THIRTEEN|FOURTEEN|FIFTEEN|SIXTEEN|SEVENTEEN|'
            r'EIGHTEEN|NINETEEN|TWENTY|TWENTY-ONE|TWENTY-TWO|TWENTY-THREE|'
            r'TWENTY-FOUR|TWENTY-FIVE|TWENTY-SIX|TWENTY-SEVEN|TWENTY-EIGHT|'
            r'TWENTY-NINE|THIRTY|THIRTY-ONE|THIRTY-TWO|THIRTY-THREE|'
            r'THIRTY-FOUR|THIRTY-FIVE|THIRTY-SIX|THIRTY-SEVEN|\d+):\s*[A-Z][A-Z\s]+)',
            doc.page_content,
            flags=re.IGNORECASE
        )
        
        # Process chapter by chapter
        current_chapter = "Unknown"
        
        for i, section in enumerate(chapter_splits):
            # Skip empty sections
            if not section.strip():
                continue
            
            # Check if this section is a chapter heading
            if re.match(r'CHAPTER\s+', section, re.IGNORECASE):
                current_chapter = extract_chapter_number(section)
                # Include the chapter heading in the text
                text_to_chunk = section
            else:
                text_to_chunk = section
            
            # Initialize the text splitter for this section
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000, 
                chunk_overlap=300,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            # Split this section into chunks
            section_chunks = text_splitter.split_text(text_to_chunk)
            
            # Create Document objects with enhanced metadata
            for chunk_text in section_chunks:
                # Skip very small chunks (likely just whitespace or artifacts)
                if len(chunk_text.strip()) < 50:
                    continue
                
                chunk_doc = Document(
                    page_content=chunk_text,
                    metadata={
                        "source": doc.metadata["source"],
                        "chapter": current_chapter  
                    }
                )
                all_chunks.append(chunk_doc)
    
    print(f"Split documents into a total of {len(all_chunks)} chunks.")
    return all_chunks
