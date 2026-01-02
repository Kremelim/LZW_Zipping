## LZW Image & Text Compressor

A Python-based implementation of the Lempel–Ziv–Welch (LZW) compression algorithm featuring an intuitive Tkinter GUI. This tool supports lossless compression and decompression for both text files and BMP images (RGB & Grayscale).

## 🚀 Features
✔ Text Compression

* Compress .txt files into binary format.

* Fast, lossless decompression back to the original text.

✔ Image Compression

* Supports .bmp images (RGB & Grayscale).

* Built-in viewer and basic editor:

* Convert images to Grayscale.

* Extract Red, Green, or Blue channels.

✔ Smart Encoding

* Variable bit-length codes based on dictionary size.

* Efficient bit-packing optimizes file size.

✔ Performance Metrics

* Entropy

* Compression Ratio (CR)

* Dictionary Size

* Real-time logging in the GUI console.

## 🛠️ Tech Stack

* Language: Python 3.x

* GUI Framework: Tkinter

* Libraries Used:

  * Pillow (PIL): Image processing & display

  * NumPy: Fast array operations

  * Collections: Frequency analysis for entropy calculation

## 📈 How It Works
### 🔑 The LZW Algorithm

This project implements LZW, a universal lossless compression method that dynamically builds a dictionary of seen sequences.

#### Encoding Process:

* Reads input and grows the dictionary with new byte/character sequences.

* Emits codes when sequences repeat.

* Packs codes using the minimum number of bits (log₂(dictionary_size)).

#### Decoding Process:

* Rebuilds the dictionary identically to the encoder.

* Fully restores the original data without needing to store the dictionary in the output file.

## 🖥️ Getting Started
✔ Prerequisites

* Make sure Python and dependencies are installed:
```bash
pip install pillow numpy
```
✔ Running the Application

* Start the GUI with:
```bash
python LZW.py
```
## 📖 Usage

1. Text Compression

* Click Load Text → select a .txt file

* Click Compress Text → produces a .bin file

* To restore, click Decompress Text and choose the .bin file

2. Image Compression

* Click Load Image → select a .bmp file

* (Optional) Use:

  * Display Grayscale

  * Channel R/G/B

3. Click Compress Image

4. View the Compression Ratio displayed in the GUI

* Compression Ratio is calculated as:

![CR Formula](https://latex.codecogs.com/png.image?\dpi{150}\bg_white%20CR%20=%20\frac{Compressed%20Size}{Original%20Size})


## 📊 Technical Details

* The tool calculates Shannon Entropy of the input data:

![Entropy Formula](https://latex.codecogs.com/png.image?\dpi{150}\bg_white%20H(X)=-\sum_{i=1}^{n}P(x_i)\log_2P(x_i))

Entropy offers a theoretical limit on how much the data can be compressed—useful for comparing expected vs. actual compression performance.

---

## 📁 Project Structure
```text
├── LZW.py              # Main app (GUI + LZW logic)
├── sample.txt          # Example text file
├── *.bmp               # Sample images for testing
└── README.md           # (This file)
```
