import os
from document_classifier import DocumentClassifier
from config import SecureConfig, setup_api_key

def main():
    """Main function to run the document classifier in CLI mode"""
    try:
        # Get API key from secure storage
        config = SecureConfig()
        api_key = config.get_api_key()
        
        if not api_key:
            # Try environment variable as fallback
            api_key = os.getenv("OPENAI_API_KEY")
            
            if not api_key:
                # If no API key found, run setup
                api_key = setup_api_key()
                
            if not api_key:
                raise ValueError("No API key available. Please configure your API key.")

        # Get model version
        print("\nAvailable models:")
        print("1. o1-2024-12-17 (GPT-4 Vision specific version)")
        print("2. chatgpt-4o-latest (GPT-4 Turbo)")
        
        model_choice = input("\nSelect model (1-2, default=1): ").strip()
        model_versions = ['o1-2024-12-17', 'chatgpt-4o-latest']
        model_version = model_versions[int(model_choice) - 1] if model_choice and model_choice in '12' else 'o1-2024-12-17'

        # Get output directory
        output_dir = input("\nEnter output directory path (press Enter for current directory): ").strip()
        if not output_dir:
            output_dir = os.getcwd()
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Initialize classifier with selected model
        classifier = DocumentClassifier(api_key, output_dir, model_version)

        # Get folder path from user
        folder_path = input("\nEnter the folder path to process: ").strip()
        if not os.path.exists(folder_path):
            raise ValueError("Folder path does not exist")

        # Get batch size from user
        batch_size = input("\nEnter batch size (press Enter for default 10): ").strip()
        batch_size = int(batch_size) if batch_size else 10

        # Process folder and display results
        results_df = classifier.process_folder(folder_path, batch_size)
        print("\nClassification Results Summary:")
        print(results_df.groupby('Document Type').size())

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main()) 