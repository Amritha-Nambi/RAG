# main.py
import sys
import os
import subprocess
from pipeline import run_pipeline

def main():
    """
    The Master Control Function.
    """
    print("\n" + "="*60)
    print("HARRY POTTER RAG SYSTEM CONTROLLER")
    print("="*60 + "\n")
    
    print("What would you like to do?")
    print("  [1] Build Database (Run Extraction & Indexing)")
    print("  [2] Run Chatbot (Web Interface)")
    print("  [3] Run Evaluation (Test Accuracy)")
    print("  [4] Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        # Calls the function from pipeline.py
        run_pipeline()
        
    elif choice == '2':
        print("\nLaunching Web Interface...")
        # Uses subprocess to run the streamlit command for you
        subprocess.run(["streamlit", "run", "app.py"])
        
    elif choice == '3':
        print("\nStarting Evaluation...")
        # Runs the evaluation script
        subprocess.run(["python", "evaluate.py"])
        
    elif choice == '4':
        print("Goodbye!")
        sys.exit()
        
    else:
        print("Invalid choice. Please try again.")
        main()

if __name__ == "__main__":
    main()