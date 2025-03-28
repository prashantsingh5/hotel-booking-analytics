import os
import sys
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import pandas as pd
from dotenv import load_dotenv

# Add the src directory to Python path to import modules
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, src_path)

# Import the HotelBookingRAG class from your existing script
from hotel_booking_rag import HotelBookingRAG

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration with flexible path handling
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG = {
    'CSV_PATH': os.path.join(BASE_DIR, 'data', 'processed', 'cleaned_hotel_bookings.csv'),
    'ANALYTICS_PDF_PATH': os.path.join(BASE_DIR, 'analytics_report', 'HotelBookingAnalytics.pdf'),
    'PERFORMANCE_REPORT_PATH': os.path.join(BASE_DIR, 'performance_evaluation', 'performance_report.json')
}

# Initialize RAG system 
rag_system = HotelBookingRAG(
    csv_path=CONFIG['CSV_PATH']
)

@app.route('/analytics', methods=['POST'])
def get_analytics_report():
    """
    Endpoint to retrieve the pre-generated analytics PDF report
    
    Request body: Optional parameters for filtering or specifying report type
    Returns: PDF file of analytics report
    """
    try:
        # Check if PDF exists
        if not os.path.exists(CONFIG['ANALYTICS_PDF_PATH']):
            return jsonify({
                'error': 'Analytics report not found',
                'message': 'Please generate the analytics report first'
            }), 404
        
        # Send the PDF file
        return send_file(
            CONFIG['ANALYTICS_PDF_PATH'], 
            mimetype='application/pdf',
            as_attachment=True,
            download_name='hotel_booking_analytics.pdf'
        )
    
    except Exception as e:
        return jsonify({
            'error': 'Failed to retrieve analytics report',
            'message': str(e)
        }), 500

@app.route('/ask', methods=['POST'])
def answer_booking_query():
    """
    Endpoint for answering booking-related questions using RAG
    
    Request body: 
    {
        "query": "What is the total revenue for July 2017?",
        "use_llm_enhancement": true  # Optional, defaults to true
    }
    
    Returns: JSON with the RAG-generated answer
    """
    try:
        # Parse request data
        request_data = request.get_json()
        query = request_data.get('query')
        use_llm_enhancement = request_data.get('use_llm_enhancement', True)
        
        # Validate query
        if not query:
            return jsonify({
                'error': 'Invalid request',
                'message': 'Query is required'
            }), 400
        
        # Use RAG system to answer the query
        answer = rag_system.answer_query(
            query, 
            use_llm_enhancement=use_llm_enhancement
        )
        
        return jsonify({
            'query': query,
            'answer': answer
        })
    
    except Exception as e:
        return jsonify({
            'error': 'Failed to process query',
            'message': str(e)
        }), 500

@app.route('/performance', methods=['GET'])
def get_performance_report():
    """
    Endpoint to retrieve the performance evaluation report
    """
    try:
        # Check if performance report exists
        if not os.path.exists(CONFIG['PERFORMANCE_REPORT_PATH']):
            return jsonify({
                'error': 'Performance report not found',
                'message': 'Please generate the performance report first'
            }), 404
        
        # Read and return the performance report
        with open(CONFIG['PERFORMANCE_REPORT_PATH'], 'r') as f:
            performance_data = json.load(f)
        
        return jsonify(performance_data)
    
    except Exception as e:
        return jsonify({
            'error': 'Failed to retrieve performance report',
            'message': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'total_bookings': len(rag_system.df),
        'system_initialized': True
    }), 200

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)