# OCR Marcación - Optical Character Recognition System

Advanced OCR (Optical Character Recognition) system designed to extract text from labels and documents with multiple image processing strategies to maximize accuracy.

## Features

- **Multiple OCR Strategies**: Combines EasyOCR and Tesseract with different preprocessing techniques
- **Intelligent Processing**: Applies adaptive filters, CLAHE, and noise reduction
- **Automatic Correction**: Intelligent correction system based on known patterns
- **Intuitive Web Interface**: Drag-and-drop interface for uploading images
- **Results Comparison**: Shows multiple results with confidence levels
- **Processing Visualization**: Displays processed images for each strategy

## Technologies

**Backend:**
- Python 3.x
- Flask (Web framework)
- OpenCV (Image processing)
- Tesseract OCR
- EasyOCR
- PIL/Pillow
- NumPy

**Frontend:**
- HTML5
- CSS3
- JavaScript (Vanilla)

## Installation

### Prerequisites

- Python 3.7 or higher
- Tesseract OCR installed on the system

**Install Tesseract:**

**Windows:**
\`\`\`bash
# Download and install from: https://github.com/UB-Mannheim/tesseract/wiki
\`\`\`

**macOS:**
\`\`\`bash
brew install tesseract
\`\`\`

**Linux (Ubuntu/Debian):**
\`\`\`bash
sudo apt-get update
sudo apt-get install tesseract-ocr
\`\`\`

### Project Installation

1. **Clone the repository:**
\`\`\`bash
git clone https://github.com/CFrancoChavez/OCR-Marcacion.git
cd OCR-Marcacion
\`\`\`

2. **Create virtual environment:**
\`\`\`bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
\`\`\`

3. **Install dependencies:**
\`\`\`bash
pip install -r requirements.txt
\`\`\`

4. **Run the application:**
\`\`\`bash
python app.py
\`\`\`

5. **Open in browser:**
\`\`\`
http://localhost:5000
\`\`\`

## Usage

1. **Upload Image**: Drag an image or click to select a file
2. **Automatic Processing**: The system applies multiple OCR strategies
3. **Review Results**: Compare results from different strategies
4. **Best Result**: The system automatically highlights the result with highest confidence

## Processing Strategies

The system implements multiple preprocessing strategies:

- **Original Unprocessed**: Grayscale image without modifications
- **Gentle Processing**: Bilateral filter and light contrast adjustment
- **Minimal Processing**: Only grayscale with minimal adjustment
- **Soft CLAHE**: Adaptive local contrast enhancement
- **Adaptive Binarization**: Gaussian adaptive thresholding
- **Otsu Binarization**: Automatic Otsu thresholding
- **Soft Morphology**: Morphological operations for noise cleaning

## Project Structure

\`\`\`
OCR-Marcacion/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Web interface
├── static/               # Static files (CSS, JS, images)
└── uploads/              # Temporary folder for uploaded images
\`\`\`

## Intelligent Correction

The system includes an intelligent correction module that:
- Detects common OCR error patterns
- Applies context-based corrections
- Uses sequence similarity to validate results
- Maintains a dictionary of known corrections

## Use Cases

- Text extraction from product labels
- Reading manufacturing and expiration dates
- Batch code processing
- Digitization of printed documents
- Data entry automation

## Future Improvements

- [ ] Batch processing support
- [ ] REST API for system integration
- [ ] Export results to CSV/JSON
- [ ] Custom model training
- [ ] Multi-language support
- [ ] Database integration

## Contributing

Contributions are welcome. Please:

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact

**Franco Chávez**
- Email: cfrancochavezdev@gmail.com
- LinkedIn: [Franco Chávez](https://www.linkedin.com/in/franco-chavez-548b0a56/)
- GitHub: [@CFrancoChavez](https://github.com/CFrancoChavez)
- Portfolio: [My Full Stack Portfolio](https://github.com/CFrancoChavez/My-FullStack-Portfolio)

## License

This project is available as open source under the MIT license.

---

Built with Python and Flask
