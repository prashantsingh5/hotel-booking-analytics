# Hotel Booking Analytics & Q&A System

## 🏨 Project Overview

This project develops a sophisticated Hotel Booking Analytics and Retrieval-Augmented Generation (RAG) system that provides deep insights into hotel booking data through advanced analytics and natural language querying.

### Key Features
- 📊 Comprehensive booking data analytics
- 🤖 AI-powered question-answering system
- 📈 Retrieval-Augmented Generation (RAG) with semantic search
- 🌐 RESTful API for analytics and querying
- 🔍 Detailed insights on revenue, cancellations, and booking trends

## 🛠 Prerequisites

### System Requirements
- Python 3.8+
- pip (Python package manager)
- git

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/hotel-booking-analytics.git
cd hotel-booking-analytics
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
Create a `.env` file in the project root with the following (optional) configurations:
```
GEMINI_API_KEY=your_google_generativeai_api_key
```
- The Gemini API key is optional and used for LLM answer enhancement
- If not provided, the system will still function without LLM enhancement

## 📦 Project Structure
```
hotel-booking-analytics/
│
├── data/
│   ├── hotel_bookings.csv
│   └── cleaned_hotel_bookings.csv
│
├── analytics_output/
│   └── (various analytics visualization images)
│
├── performance_evaluation/
│   └── (performance metrics and reports)
│
├── hotel_booking_rag.py     # Main RAG system implementation
├── flask_integration.py     # Flask API endpoints
├── performance_evaluation.py
├── preprocessing_script.py
└── .env
```

## 🔧 Running the Application

### Start the Flask API Server
```bash
python flask_integration.py
```
- Server will run on `http://localhost:5000`
- API Endpoints:
  - `POST /analytics`: Retrieve analytics report
  - `POST /ask`: Ask questions about hotel bookings
  - `GET /health`: System health check

### Example API Queries
You can use tools like `curl` or Postman to interact with the API:

1. Ask a Question:
```bash
curl -X POST http://localhost:5000/ask \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the total revenue for July 2017?"}'
```

2. Get Health Status:
```bash
curl http://localhost:5000/health
```

## 📊 Sample Queries
- Total revenue for specific months/years
- Cancellation rates and locations
- Hotel type counts
- Booking trends
- Average booking prices
- Market segment analysis
- Special requests and booking changes

## 🧪 Running Performance Evaluation
```bash
python performance_evaluation.py
```

## 🙏 Acknowledgements
- Dataset Source: [Kaggle Hotel Bookings Dataset](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
- Libraries: Pandas, NumPy, Sentence Transformers, FAISS, Flask