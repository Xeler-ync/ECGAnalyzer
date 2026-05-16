
# ECGAnalyzer: Advanced ECG Signal Analysis Platform

## 🚀 Quick Start

### Prerequisites
- Python 3 (Developed and tested with Python 3.12.3)
- pip package manager

### Installation
1. Clone the repository:
```bash
git clone https://github.com/Xeler-ync/ECGAnalyzer.git
cd ECGAnalyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Database Setup

### PTB-XL Database Integration

1. **Download Database**
   - Visit [PhysioNet PTB-XL](https://physionet.org/content/ptb-xl/)
   - Download the latest database release
   - Extract the archive

2. **Directory Structure**
   Ensure your project follows this structure:
   ```
   ECGAnalyzer/
   ├── PTB-XL/          # PTB-XL database
   │   ├── records100/  # 100Hz records
   │   └── records500/  # 500Hz records
   ├── test/            # algorithm test
   ├── utils/           # Utility functions
   ├── main.py          # Main application
   ├── README.md
   └── requirements.txt
   ```

## 🛠️ Basic Usage

### 1. Data Processing & Heartbeat Extraction
Run the main data pipeline to filter, detect R-peaks, and extract normalized heartbeats based on SCP codes or superclasses:
```bash
# Filter by SCP code (e.g., LVH, NORM)
python ./tools/data.py --scp_codes LVH --min_value 100
python ./tools/data.py --scp_codes NORM --min_value 100

# Filter by Superclass (e.g., MI)
python ./tools/data.py --superclass MI

# Generate OTHERS dataset (exclude positive and pure negative)
python ./tools/data.py --others --positive-code LVH --negative-code NORM
python ./tools/data.py --others --positive-is-superclass --positive-superclass MI --negative-code NORM
```

### 2. Model Training & Evaluation
Train Random Forest classifiers using the extracted normalized heartbeats:
```bash
# Binary classification: LVH vs NORM
python ./train/train_lvh_norm_rf.py

# Multi-class classification: LVH vs NORM vs OTHERS
python ./train/train_lvh_norm_other_rf.py

# Binary classification: MI vs NORM
python ./train/train_mi_norm_rf.py

# Multi-class classification: MI vs NORM vs OTHERS
python ./train/train_mi_norm_other_rf.py
```