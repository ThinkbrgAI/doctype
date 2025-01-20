# Document Classification System

A vision-based document classification system using GPT-4 Vision API.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run setup:
   ```bash
   python setup_environment.py
   ```

3. Place PDF files in `input/pdfs/` directory

4. Run classifier:
   ```bash
   python run_classifier.py
   ```

## Directory Structure

- `input/pdfs/`: Place PDF files here for classification
- `output/reports/`: Classification results
- `output/logs/`: System logs
- `config/`: Configuration files
- `samples/`: Sample document categories
- `temp/`: Temporary processing files

## Security

API keys are stored securely in the user's home directory:
- Encrypted configuration: `~/.doctype/config.enc`
- Encryption key: `~/.doctype/.key`
