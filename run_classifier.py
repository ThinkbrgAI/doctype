import sys

def main():
    """Launch the document classifier in GUI mode by default"""
    try:
        # Check if --cli flag is used
        if len(sys.argv) > 1 and sys.argv[1] == '--cli':
            from run_classifier_cli import main as cli_main
            return cli_main()
        else:
            from document_classifier_gui import main as gui_main
            return gui_main()
    except ImportError as e:
        print(f"Error: {str(e)}")
        print("\nMake sure all required files are present and dependencies are installed:")
        print("1. run_classifier_cli.py - for command line interface")
        print("2. document_classifier_gui.py - for graphical interface")
        print("\nTry running: pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    exit(main()) 