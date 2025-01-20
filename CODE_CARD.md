# Document Classification System
A Python-based system for automatically classifying construction documents using OpenAI's GPT models.

## Core Features
- Batch processing of PDF documents
- Two AI model options:
  1. GPT-4 Vision (o1-2024-12-17) - Specialized for document analysis
  2. GPT-4 Turbo (chatgpt-4o-latest) - Latest general-purpose model
- Secure API key management
- Progress tracking and error handling
- Detailed logging and Excel report generation

## Architecture

### 1. Core Components
```
├── document_classifier.py    # Main classification engine
├── run_classifier.py        # CLI interface
├── config.py               # Configuration and API key management
├── setup_environment.py    # Environment setup
└── create_folder_structure.py  # Directory structure creation
```

### 2. Directory Structure
```
A:/doctype/
├── input/
│   ├── pdfs/         # Place PDFs here for classification
│   └── test_files/   # Test documents
├── output/
│   ├── reports/      # Classification results (Excel)
│   └── logs/         # System logs
├── config/           # Project configuration
└── samples/          # Sample document categories
```

## Document Categories
1. Plans & Specifications
2. Key Dates and Schedules
3. Contracts and Changes
4. Meeting Minutes
5. Pay Applications
6. Daily Reports
7. Inspection Reports
8. Documentation
9. Miscellaneous

## Security Features
- Encrypted API key storage
- Secure file permissions
- API key stored in user's home directory (~/.doctype/)
- Environment variable support

## Usage

### 1. Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
python setup_environment.py

# Configure API key
python run_classifier.py
```

### 2. Running Classifications

#### GUI Mode (Default)
```bash
python run_classifier.py
```

#### CLI Mode (Optional)
```bash
python run_classifier.py --cli
```

### 3. Output
- Excel report with classifications
- Confidence scores
- Processing logs
- Error reports

## Technical Details

### AI Models
1. **GPT-4 Vision (o1-2024-12-17)**
   - Specialized for document analysis
   - Can process visual document features
   - Best for complex document layouts

2. **GPT-4 Turbo (chatgpt-4o-latest)**
   - Latest general-purpose model
   - Faster processing
   - Good for text-heavy documents

### Processing Pipeline
1. Document Loading
   - PDF validation
   - File type checking
   - Base64 encoding

2. Classification
   - Model-specific processing
   - Confidence scoring
   - Category mapping

3. Results Management
   - Excel report generation
   - Error handling
   - Progress tracking

### Dependencies
```
fastapi>=0.68.0
uvicorn>=0.15.0
python-multipart>=0.0.5
python-dotenv>=0.19.0
openai>=1.0.0
pdf2image>=1.16.3
python-magic>=0.4.27
pandas>=1.5.3
Pillow>=9.5.0
openpyxl>=3.1.2
cryptography>=41.0.0
```

## Error Handling
- PDF validation errors
- API connection issues
- Model processing errors
- File system errors
- Batch processing recovery

## Performance
- Batch processing for efficiency
- Configurable batch sizes
- Progress tracking
- Intermediate result saving

## Logging
- Detailed processing logs
- Error tracking
- Performance metrics
- Classification statistics

## Future Enhancements
- Additional model support
- Enhanced error recovery
- Automated testing
- Performance optimization
- UI improvements 