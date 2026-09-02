# IndoBERT News Classification

A comprehensive machine learning project for **Indonesian sports news classification** using IndoBERT and PyTorch. This project demonstrates end-to-end NLP pipeline including web scraping, data preprocessing, exploratory data analysis, and deep learning model training.

![Badge](https://img.shields.io/badge/Language-Python-blue) ![Badge](https://img.shields.io/badge/Framework-PyTorch-red) ![Badge](https://img.shields.io/badge/Model-IndoBERT-green) ![Badge](https://img.shields.io/badge/License-MIT-yellow)

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Web Scraping](#1-web-scraping)
  - [2. Data Preprocessing](#2-data-preprocessing)
  - [3. Model Training](#3-model-training)
- [Data Analysis](#data-analysis)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Project Overview

This project implements a **multi-class text classification system** for Indonesian sports news articles. The model classifies news into five categories based on content:

- **Liga Inggris** (English Premier League)
- **Liga Italia** (Italian Serie A)
- **Liga Spanyol** (Spanish La Liga)
- **Liga Indonesia** (Indonesian Football League)
- **Non-Sepak Bola** (Non-Football Sports)

The project uses **IndoBERT**, a BERT model pre-trained on Indonesian text, combined with PyTorch for transfer learning and fine-tuning.

---

## 📊 Dataset

### Data Collection
- **Sources**: 3 major Indonesian media outlets
  - Detik Sports (sport.detik.com)
  - Kompas Sports (bola.kompas.com)
  - Liputan6 Sports (liputan6.com)

- **Total Articles**: 2,276 news articles
- **Collection Method**: Web scraping with URL-based labeling
- **Date Range**: Recent sports news articles

### Dataset Statistics

| Label | Count | Percentage |
|-------|-------|-----------|
| Liga Indonesia | 367 | 16.12% |
| Liga Inggris | 376 | 16.52% |
| Liga Italia | 386 | 16.96% |
| Liga Spanyol | 386 | 16.96% |
| Non-Sepak Bola | 761 | 33.44% |
| **Total** | **2,276** | **100%** |

### Media Distribution

| Media | Count | Percentage |
|-------|-------|-----------|
| Detik | 1,052 | 46.22% |
| Kompas | 992 | 43.59% |
| Liputan6 | 232 | 10.19% |
| **Total** | **2,276** | **100%** |

---

## 📁 Project Structure

```
IndoBERT-News-Classification/
├── README.md                          # Project documentation
├── 1a_scrapping.ipynb                # Web scraping notebook
├── 1.ipynb                            # Main analysis & modeling notebook
├── output/                            # Generated outputs directory
│   └── dataset_final.csv             # Final processed dataset
├── checkpoints/                       # Model checkpoints (during scraping)
├── dist_label.png                    # Label distribution visualization
├── dist_media.png                    # Media source distribution
├── crosstab_media_label.png          # Cross-tabulation heatmap
├── text_length.png                   # Text length statistics
├── top20_words_before.png            # Top 20 words before preprocessing
└── wordcloud_labels.png              # Word cloud by label
```

---

## ✨ Features

### 1. **Web Scraping Pipeline** (`1a_scrapping.ipynb`)
- Multi-source web scraping from 3 Indonesian media outlets
- 21 configured sources with targeted distribution per label/media
- Robust HTTP client with:
  - Random User-Agent rotation
  - Retry mechanism (exponential backoff)
  - Connection pooling with `requests.Session()`
  - Error handling and logging
- Dual-method content extraction:
  - Primary: `trafilatura` (specialized news extractor)
  - Fallback: `BeautifulSoup` (HTML parsing)
- Pagination handling for all media sources
- URL deduplication and validation
- Data quality checks and checkpoint saving

### 2. **Data Preprocessing & EDA** (`1.ipynb`)
- **Missing Value Analysis**: Detection and handling of null values
- **Duplicate Detection**: Identifying and removing duplicate articles
- **Exploratory Data Analysis (EDA)**:
  - Label distribution visualization
  - Media source distribution
  - Cross-tabulation analysis (media × label)
  - Text length statistics (character & word count)
  - Top 20 words frequency analysis
  - Word clouds per label
- **Text Preprocessing**:
  - Indonesian stopword removal (Sastrawi)
  - Text normalization and cleaning
  - Stemming using Sastrawi stemmer
  - Case normalization

### 3. **Model Architecture**
- **Base Model**: `indobenchmark/indobert-base-p1`
- **Architecture**: Transformer-based BERT with custom classification head
- **Features**:
  - Fine-tuning on Indonesian news corpus
  - 1-Stage Model: Direct classification from article text
  - 2-Stage Model: Alternative hierarchical approach
  - GPU acceleration support (CUDA)

### 4. **Model Training Pipeline**
- Adam optimizer with learning rate scheduling
- Batch processing with DataLoader
- Cross-entropy loss for multi-class classification
- Training/validation/test split
- Early stopping and checkpoint management
- Comprehensive metrics evaluation

### 5. **Evaluation Metrics**
- Accuracy Score
- Precision, Recall, F1-Score
- Confusion Matrix Analysis
- Classification Report with per-class metrics
- Support for class-wise performance analysis

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA 12.1 (for GPU acceleration, optional but recommended)
- pip or conda

### Step 1: Clone the Repository
```bash
git clone https://github.com/RichelleMarvela/IndoBERT-News-Classification.git
cd IndoBERT-News-Classification
```

### Step 2: Create Virtual Environment
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n indobert python=3.10
conda activate indobert
```

### Step 3: Install Dependencies
```bash
# Core dependencies
pip install torch transformers scikit-learn pandas numpy

# Data processing
pip install beautifulsoup4 lxml trafilatura requests

# NLP utilities
pip install nltk sastrawi fake-useragent

# Visualization
pip install matplotlib seaborn wordcloud

# Development
pip install tqdm jupyter notebook ipython

# Or install all at once
pip install -r requirements.txt  # If requirements.txt exists
```

### Step 4: Download NLTK Resources
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

---

## 📖 Usage

### 1. Web Scraping

Open `1a_scrapping.ipynb` and follow these steps:

```python
# The notebook includes:
# 1. Library imports and configuration
# 2. Source configuration (21 sources from 3 media)
# 3. HTTP client setup with retry mechanism
# 4. URL extraction per media type
# 5. Article content extraction
# 6. Dataset assembly and validation
# 7. CSV export to output/dataset_final.csv
```

**Configuration Targets**:
- Liga Inggris: 400 articles (100 detik, 250 kompas, 50 liputan6)
- Liga Italia: 400 articles (100 detik, 250 kompas, 50 liputan6)
- Liga Spanyol: 400 articles (100 detik, 250 kompas, 50 liputan6)
- Liga Indonesia: 400 articles (100 detik, 250 kompas, 50 liputan6)
- Non-Sepak Bola: 800 articles (652 detik, 60 kompas, 88 liputan6)

**Output**: `output/dataset_final.csv` with columns:
- `text`: Article full text (title + content)
- `media`: Source media (detik/kompas/liputan6)
- `label`: Classification label
- `url`: Source article URL
- `date`: Publication date (if available)

### 2. Data Preprocessing & Analysis

Open `1.ipynb` and execute:

```python
# Section 1: Dataset Loading
# - Load dataset_final.csv
# - Display shape and columns
# - Show sample data (head 5 rows)

# Section 2: Data Quality Checks
# - Missing value analysis with visualization
# - Duplicate detection based on article text
# - Data cleaning and validation

# Section 3: Exploratory Data Analysis (EDA)
# - Label distribution (bar chart & pie chart)
# - Media distribution (bar chart & pie chart)
# - Cross-tabulation media vs label (heatmap)
# - Text length statistics (histogram & box plot)
# - Top 20 most frequent words (bar chart)
# - Word clouds per label (colorful visualizations)

# Section 4: Text Preprocessing
# - Tokenization
# - Stopword removal (Indonesian)
# - Stemming
# - Normalization
```

**Outputs Generated**:
- `output/eda_missing_value.png`
- `output/eda_label_distribution.png`
- `output/eda_media_distribution.png`
- `output/eda_crosstab_media_label.png`
- `output/eda_text_length_distribution.png`
- `output/eda_top20_words.png`
- `output/eda_wordcloud_per_label.png`

### 3. Model Training (In Development)

```python
# Placeholder for model training section:
# - Train/val/test split (typically 70/15/15)
# - Tokenizer initialization with IndoBERT
# - Custom Dataset class for PyTorch DataLoader
# - Model definition (BERT + classification head)
# - Training loop with validation
# - Hyperparameter tuning
# - Evaluation and metrics reporting
# - Model saving and inference
```

---

## 📈 Data Analysis

### Label Distribution
The dataset shows **imbalanced classes** with `non_sepak_bola` being the largest class at 33.44%, while the four football league categories are more balanced (16-17% each).

**Visualization**: See `dist_label.png`

### Media Source Distribution
- **Detik**: 46.22% (majority contributor)
- **Kompas**: 43.59% (significant contributor)
- **Liputan6**: 10.19% (supplementary source)

**Visualization**: See `dist_media.png`

### Cross-Media-Label Distribution
Shows how articles are distributed across media sources for each label category.

**Visualization**: See `crosstab_media_label.png`

### Text Statistics
- **Character Count**: Varies by label, typically 1,500-4,000 characters per article
- **Word Count**: Average 200-400 words per article
- **Distribution**: Relatively normal distribution with some outliers

**Visualizations**: See `text_length.png`

### Most Frequent Words
Top 20 words before preprocessing show common sports terminology and article-specific keywords.

**Visualization**: See `top20_words_before.png`

### Word Clouds
Label-specific word clouds provide visual representation of dominant terms in each category.

**Visualization**: See `wordcloud_labels.png`

---

## 🧠 Model Architecture

### IndoBERT Base Configuration
```
Model: indobenchmark/indobert-base-p1
├── Vocabulary: ~32,000 tokens (Indonesian)
├── Hidden Layers: 12
├── Attention Heads: 12
├── Hidden Size: 768
├── Intermediate Size: 3,072
├── Max Position Embeddings: 512
└── Type Vocab Size: 2
```

### Classification Head
```
Input: [CLS] Token Embedding (768-dim)
  ↓
Dropout (p=0.1)
  ↓
Dense Layer: 768 → 256 (ReLU activation)
  ↓
Dropout (p=0.1)
  ↓
Dense Layer: 256 → 5 (Output classes)
  ↓
Softmax
Output: Class Probabilities
```

### Training Configuration
- **Optimizer**: Adam (lr=2e-5)
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 32
- **Epochs**: 10-20 (with early stopping)
- **Device**: GPU (CUDA) if available, else CPU

---

## 📊 Results

### Expected Performance Metrics
Based on similar BERT-based Indonesian text classification tasks:

- **Overall Accuracy**: 85-92%
- **Macro-averaged F1**: 82-88%
- **Weighted F1**: 85-91%

### Per-Class Performance
Performance may vary by class due to imbalanced dataset. Fine-tuning strategies like:
- Weighted loss function
- Class-balanced sampling
- Focal loss
- Threshold adjustment

can improve minority class performance.

---

## 🛠 Technologies Used

### Core Libraries
| Library | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.5.1+ | Deep learning framework |
| Transformers | 4.x | Pre-trained BERT models |
| Scikit-learn | 1.x | ML utilities & metrics |
| Pandas | 2.x | Data manipulation |
| NumPy | 1.x | Numerical computing |

### NLP & Text Processing
| Library | Purpose |
|---------|---------|
| NLTK | Tokenization & stopwords |
| Sastrawi | Indonesian stemming |
| BeautifulSoup4 | HTML parsing |
| Trafilatura | News content extraction |
| Lxml | XML/HTML processing |

### Web Scraping
| Library | Purpose |
|---------|---------|
| Requests | HTTP requests |
| Fake-UserAgent | User-Agent rotation |
| Beautiful Soup | HTML parsing |
| Trafilatura | Article extraction |

### Data Visualization
| Library | Purpose |
|---------|---------|
| Matplotlib | Plot generation |
| Seaborn | Statistical visualizations |
| WordCloud | Word cloud generation |

### Development Tools
| Tool | Purpose |
|------|---------|
| Jupyter Notebook | Interactive development |
| Git | Version control |
| Python 3.10+ | Programming language |

---

## 📦 Requirements

### System Requirements
- **RAM**: 8GB minimum (16GB recommended)
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **Disk Space**: 10GB (including model weights and dataset)

### Python Packages
```
torch>=2.0.0
transformers>=4.30.0
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
beautifulsoup4>=4.10.0
lxml>=4.9.0
trafilatura>=1.6.0
requests>=2.28.0
fake-useragent>=1.1.0
nltk>=3.8.0
sastrawi>=1.0.0
matplotlib>=3.5.0
seaborn>=0.12.0
wordcloud>=1.9.0
tqdm>=4.64.0
jupyter>=1.0.0
ipython>=8.0.0
```

---

## 👤 Contact

**Author**: Richelle Marvela  
**GitHub**: [@RichelleMarvela](https://github.com/RichelleMarvela)  
**Repository**: [IndoBERT-News-Classification](https://github.com/RichelleMarvela/IndoBERT-News-Classification)

For questions or suggestions, please open an issue on GitHub.

---

## 📚 Additional Resources

### IndoBERT References
- [IndoBERT on Hugging Face](https://huggingface.co/indobenchmark/indobert-base-p1)
- [IndonesianBERT GitHub](https://github.com/indobenchmark/indobert)

### BERT & Transfer Learning
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers/)

### Indonesian NLP
- [Sastrawi Stemmer](https://github.com/har07/PySastrawi)
- [NLTK Indonesian Support](https://www.nltk.org/)

### PyTorch Resources
- [PyTorch Official Documentation](https://pytorch.org/docs/)
- [PyTorch Tutorial](https://pytorch.org/tutorials/)

---

## 📝 Project Timeline

- **Phase 1**: Web scraping infrastructure & data collection ✅
- **Phase 2**: Data preprocessing & exploratory analysis ✅
- **Phase 3**: Model training & evaluation 🔄
- **Phase 4**: Hyperparameter tuning & optimization ⏳
- **Phase 5**: Deployment & inference API ⏳
- **Phase 6**: Documentation & case studies ⏳

---

## 🎓 Project Motivation

This project demonstrates a complete machine learning pipeline for Indonesian text classification, specifically designed for sports news categorization. It serves as:

- A practical example of transfer learning with BERT
- A guide for NLP practitioners working with Indonesian language
- A demonstration of end-to-end ML project management
- A resource for understanding multi-source web scraping
- A template for similar text classification tasks

