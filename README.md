<img width="1889" height="971" alt="image" src="https://github.com/user-attachments/assets/57f8b0c1-8891-4fa0-a8ac-e5031d6cd570" />
<img width="1885" height="967" alt="image" src="https://github.com/user-attachments/assets/e02d1a0d-8132-428d-8372-c3f4874d4618" />
<img width="1892" height="959" alt="image" src="https://github.com/user-attachments/assets/ffd8c1c0-9132-4b1d-af40-5e3529e5d1f5" />


# Stock Price Prediction System

A comprehensive full-stack web application for real-time stock price analysis and prediction using 7 machine learning algorithms, supporting both US and Indian markets with live data fetching, historical analysis, and interactive visualization.

## Key Features

- **7 ML Algorithms**: Linear Regression, Decision Tree, Random Forest, SVM (basic models) + KNN, ARIMA, Autoencoder (advanced models)
- **Real-time Data**: US stocks via Finnhub API, Indian stocks via Upstox API with permanent storage fallback
- **Modern Dashboard**: React 18 + TypeScript + Tailwind CSS with interactive Recharts for 5-year historical analysis
- **1000+ Stocks**: 1,001 total stocks available (501 US + 500 Indian), 936 used for training after filtering insufficient data
- **Currency Support**: Real-time USD/INR conversion via forex-python
- **Smart Training**: Percentage-based predictions with proper price conversion and confidence scoring
- **Standalone Trainers**: Independent training scripts for each model with progress tracking

## Currency Conversion

Automatic USD/INR conversion for all stocks:
- **Sources**: Live forex APIs → Cached rates → Static rate (83.5 USD/INR)
- **Display**: Original currency + converted price shown for all stocks

## Technology Stack

**Backend**: Flask 2.3.3, Python 3.8+, TensorFlow 2.20, scikit-learn 1.5.2, statsmodels 0.14.4, pandas, numpy  
**Frontend**: React 18.3.1, TypeScript, Vite 6.4.0, Tailwind CSS, Radix UI, Recharts 2.15.2  
**APIs**: Finnhub (US stocks), Upstox (Indian stocks), yfinance (historical data)

## Offline Mode First

**The system works completely offline without any API keys!** This is the recommended way to get started.

### What Works Offline
- **Stock Information**: 500 Indian + 501 US stocks (1,001 total) from permanent directory
- **Historical Charts**: Complete 5-year OHLCV data (2020-2024)
- **ML Predictions**: All trained models work with offline data (trained on 936 stocks with sufficient data)
- **Search**: Full-text search across 1,001 stocks
- **Technical Indicators**: 38 indicators calculated from historical data
### What doesn't Work Offline
- **Live Prices**: Requires API keys (Finnhub for US, Upstox for India)

## Installation Guide

### Prerequisites
- **Python**: 3.8 or higher
- **Node.js**: 16 or higher
- **Git**: For cloning repository
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 15GB free space

### Windows Installation

#### 1. Install Python
```powershell
# Using winget (Windows Package Manager)
winget install Python.Python.3.11

# Or download from python.org
# Verify installation
python --version
```

#### 2. Install Node.js
```powershell
# Using winget
winget install OpenJS.NodeJS

# Verify installation
node --version
npm --version
```

#### 3. Clone and Setup
```powershell
# Clone repository
git clone https://github.com/ankit02327/stock-price.git

cd stock-price

# Backend setup
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Create empty .env file (REQUIRED)
type nul > .env

# Frontend setup (new terminal)
cd ../frontend
npm install
```

### macOS Installation

#### 1. Install Homebrew (if not installed)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### 2. Install Python and Node.js
```bash
# Install Python and Node.js
brew install python@3.11 node

# Verify installations
python3 --version
node --version
npm --version
```

#### 3. Clone and Setup
```bash
# Clone repository
git clone https://github.com/ankit02327/stock-price.git

cd stock-price

# Backend setup
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create empty .env file (REQUIRED)
touch .env

# Frontend setup (new terminal)
cd ../frontend
npm install
```

### Linux Installation (Ubuntu/Debian)

#### 1. Update Package List
```bash
sudo apt update
```

#### 2. Install Python and Node.js
```bash
# Install Python and pip
sudo apt install python3.11 python3.11-pip python3.11-venv

# Install Node.js
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Verify installations
python3 --version
node --version
npm --version
```

#### 3. Clone and Setup
```bash
# Clone repository
git clone https://github.com/ankit02327/stock-price.git

cd stock-price

# Backend setup
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create empty .env file (REQUIRED)
touch .env

# Frontend setup (new terminal)
cd ../frontend
npm install
```

## Quick Start

### 1. Start Backend
```bash
cd backend
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
python main.py
```

### 2. Start Frontend (New Terminal)
```bash
cd frontend
npm run dev
```

Access: http://localhost:5173 (Backend: http://localhost:5000)

## Data Structure

**Indian Stocks (500)**: Use ISIN codes (INExxxxxxxx) at `permanent/ind_stocks/`  
**US Stocks (501)**: Use ticker symbols (AAPL, MSFT, etc.) at `permanent/us_stocks/`

```bash
# Train: python backend/training/basic_models/{model}/trainer.py
# Status: python status.py
# Details: See [Training Guide](documentation/TRAINING.md)
```

## Optional: Live Data Setup

Add API keys to `backend/.env` for real-time data (Finnhub for US, Upstox for India).

**Guides**: [Offline Mode](documentation/OFFLINE_MODE.md) | [Training Guide](documentation/TRAINING.md) | [Backend API](backend/README.md) | [Upstox Integration](documentation/UPSTOX_INTEGRATION.md)

## System Status

Check: `python status.py` | Test API: `curl http://localhost:5000/health`

**Working**: Data fetching, 5-year historical charts, search (1000+ stocks), currency conversion, interactive dashboard, 38 indicators, ML predictions (4/7 models trained)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Backend won't start | Verify `permanent/` directory exists and create empty `.env` file |
| Frontend build errors | Run `rm -rf node_modules package-lock.json && npm install` |
| Python import errors | Activate venv: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (macOS/Linux) |
| Out of memory | Close other apps, ensure 8GB+ RAM, train one model at a time |

## Important Notes

- **Educational Use**: This application is for learning and research purposes only
- **Investment Disclaimer**: Stock predictions are inherently uncertain and should not be used as sole investment advice
- **API Limits**: Finnhub (60 calls/min free), Upstox tokens expire daily
- **Offline First**: System designed to work without internet or API keys

## Quick Commands

```bash
python status.py  # Check status
python main.py    # Start backend
npm run dev       # Start frontend

```
<div align="center">

![alt text](https://github.com/user-attachments/assets/cde2229a-a14a-461d-8d41-b7d78dd30394)

</div>
