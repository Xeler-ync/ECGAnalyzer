
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
