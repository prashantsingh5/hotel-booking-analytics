import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import json
import re
import torch
import warnings
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv

class HotelBookingRAG:
    def __init__(self, csv_path=None, batch_size=1000):
        # Load environment variables
        load_dotenv()
        
        # Suppress specific warnings
        warnings.filterwarnings('ignore', category=UserWarning)
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        
        # Determine CSV path
        if csv_path is None:
            # Default to data folder in the current project structure
            csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'cleaned_hotel_bookings.csv')
        
        # Initialize Gemini LLM if API key is available
        self.llm = None
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if gemini_api_key:
            try:
                genai.configure(api_key=gemini_api_key)
                self.llm = genai.GenerativeModel('gemini-1.5-pro-latest')
            except Exception as e:
                print(f"Error initializing Gemini LLM: {e}")
        
        # Load and preprocess the dataset
        self.df = pd.read_csv(csv_path)
        self.batch_size = batch_size
        
        # Preprocess and clean the data
        self.preprocess_data()
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create vector embeddings and index
        self.create_vector_index()
        
        # Precompute detailed insights
        self.precompute_detailed_insights()

    def preprocess_data(self):
        # Convert date columns to datetime
        date_columns = ['arrival_date', 'reservation_status_date']
        for col in date_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')

        # Convert numeric columns
        numeric_columns = ['lead_time', 'adults', 'children', 'babies', 
                        'previous_cancellations', 'previous_bookings_not_canceled',
                        'booking_changes', 'total_of_special_requests', 
                        'total_nights', 'adr']
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # Fill NaN values based on column type
        for col in self.df.columns:
            if self.df[col].dtype == 'O':  # Object type (string)
                self.df[col].fillna('Unknown', inplace=True)
            else:
                self.df[col].fillna(0, inplace=True)  # Use 0 for numeric and datetime columns

    def create_vector_index(self):
            # Create text representations for embeddings with batch processing
            def create_text_representation(row):
                return (f"Booking for {row.get('hotel', 'Unknown')} type hotel. "
                        f"Country: {row.get('country', 'Unknown')}. "
                        f"Arrival date: {row.get('arrival_date', 'Unknown')}. "
                        f"Reservation status: {row.get('reservation_status', 'Unknown')}. "
                        f"Market segment: {row.get('market_segment', 'Unknown')}.")
            
            # Generate text representations
            self.text_representations = self.df.apply(create_text_representation, axis=1)
            
            # Determine embedding dimension
            sample_embedding = self.embedding_model.encode([self.text_representations.iloc[0]])
            embedding_dim = sample_embedding.shape[1]
            
            # Create FAISS index
            self.index = faiss.IndexFlatL2(embedding_dim)
            
            # Batch processing for embeddings
            for i in range(0, len(self.text_representations), self.batch_size):
                batch = self.text_representations.iloc[i:i+self.batch_size].tolist()
                
                # Use GPU if available
                if torch.cuda.is_available():
                    # Perform encoding on GPU
                    with torch.no_grad():
                        batch_embeddings = self.embedding_model.encode(batch, device='cuda')
                else:
                    # CPU fallback
                    batch_embeddings = self.embedding_model.encode(batch)
                
                # Add batch embeddings to FAISS index
                self.index.add(batch_embeddings)
    
    def precompute_detailed_insights(self):
        # Enhanced insights computation
        self.insights = {
            'cancellation_summary': self.compute_cancellation_summary(),
            'revenue_by_month': self.compute_revenue_by_month(),
            'cancellation_by_location': self.compute_cancellation_by_location(),
            'revenue_by_market_segment': self.compute_revenue_by_market_segment(),
            'booking_trends': self.compute_booking_trends(),
            'hotel_stats': self.compute_hotel_stats()
        }
    
    def compute_revenue_by_month(self):
        # Compute monthly revenue with details
        monthly_revenue = {}
        
        # Group by month and compute total revenue
        monthly_group = self.df.groupby(pd.Grouper(key='arrival_date', freq='M'))
        for month, group in monthly_group:
            monthly_revenue[month.strftime('%B %Y')] = {
                'total_bookings': len(group),
                'total_revenue': (group['adr'] * group['total_nights']).sum(),
                'average_daily_rate': group['adr'].mean()
            }
        
        return monthly_revenue
    
    def compute_cancellation_by_location(self):
        # Detailed cancellation analysis by location
        cancellation_by_location = {}
        
        # Group by country and compute cancellation details
        country_group = self.df[self.df['is_canceled'] == 1].groupby('country')
        for country, group in country_group:
            cancellation_by_location[country] = {
                'total_cancellations': len(group),
                'cancellation_rate': len(group) / len(self.df[self.df['country'] == country]) * 100,
                'market_segments': group['market_segment'].value_counts().to_dict()
            }
        
        # Sort by total cancellations in descending order
        return dict(sorted(cancellation_by_location.items(), 
                           key=lambda x: x[1]['total_cancellations'], 
                           reverse=True))
    
    def compute_cancellation_summary(self):
        # Compute cancellation rates and insights
        total_bookings = len(self.df)
        canceled_bookings = self.df[self.df['is_canceled'] == 1]
        
        return {
            'total_bookings': total_bookings,
            'canceled_bookings': len(canceled_bookings),
            'cancellation_rate': len(canceled_bookings) / total_bookings * 100,
            'cancellation_by_country': self.df[self.df['is_canceled'] == 1]['country'].value_counts().to_dict(),
            'cancellation_by_market_segment': self.df[self.df['is_canceled'] == 1]['market_segment'].value_counts().to_dict()
        }
    
    def compute_revenue_by_market_segment(self):
        # Compute revenue by market segment
        revenue_by_segment = self.df.groupby('market_segment')['adr'].agg(['mean', 'sum']).to_dict()
        return revenue_by_segment
    
    def compute_booking_trends(self):
        # Compute booking trends
        return {
            'bookings_by_week': self.df.groupby('arrival_date_week_number').size().to_dict(),
            'bookings_by_country': self.df.groupby('country').size().to_dict(),
            'avg_lead_time': self.df['lead_time'].mean()
        }
    
    def compute_hotel_stats(self):
        # Compute additional hotel-related statistics
        return {
            'hotel_types': self.df['hotel'].value_counts().to_dict(),
            'avg_special_requests': self.df['total_of_special_requests'].mean(),
            'avg_booking_changes': self.df['booking_changes'].mean()
        }
    
    def semantic_search(self, query, top_k=5):
        # Embed the query
        query_embedding = self.embedding_model.encode([query])
        
        # Search in the index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Return matched rows
        return self.df.iloc[indices[0]]
    
    def enhance_answer_with_llm(self, query, original_answer):
        """
        Enhance the original answer using Gemini LLM to make it more user-friendly
        
        Args:
            query (str): Original user query
            original_answer (str): Answer generated by the RAG system
        
        Returns:
            str: Enhanced, more user-friendly answer
        """
        if not self.llm:
            return original_answer
        
        try:
            # Prepare the prompt for the LLM
            enhancement_prompt = f"""
            Context: You are helping a user understand hotel booking analytics.
            
            Original Query: {query}
            Original Technical Answer: {original_answer}
            
            Please rewrite the answer in a clear, conversational, and easy-to-understand manner:
            - Explain any technical terms
            - Provide context for the numbers
            - Make the language more approachable for a non-technical user
            - Highlight the most important insights
            - Keep the core information from the original answer intact
            """
            
            # Generate enhanced response
            response = self.llm.generate_content(enhancement_prompt)
            
            # Return the enhanced answer
            return response.text
        except Exception as e:
            print(f"Error enhancing answer with LLM: {e}")
            return original_answer
        
    def answer_query(self, query, use_llm_enhancement=True):
        # Existing query handling logic
        query = query.lower()

        # Expanded query patterns with more comprehensive handlers
        query_patterns = [
            # Revenue and financial queries with date support
            (r'total revenue.*(\d{4})', self.handle_monthly_revenue),
            (r'revenue for.*(\w+ \d{4})', self.handle_monthly_revenue),
            
            # Cancellation location queries
            (r'locations.*highest.*cancellation', self.handle_cancellation_locations),
            (r'cancellation.*by location', self.handle_cancellation_locations),
            
            # Previous existing patterns
            (r'total revenue.*market segment', self.handle_revenue_by_market_segment),
            (r'revenue by segment', self.handle_revenue_by_market_segment),
            
            (r'cancellation.*rate', self.handle_cancellation_summary),
            (r'cancel.*summary', self.handle_cancellation_summary),
            
            (r'booking trends', self.handle_booking_trends),
            (r'trend', self.handle_booking_trends),
            
            (r'average price', self.handle_average_price),
            (r'avg price', self.handle_average_price),
            
            (r'total count.*hotel', self.handle_hotel_count),
            (r'hotel types', self.handle_hotel_count),
            (r'how many.*hotel', self.handle_hotel_count),
            (r'number of.*hotel', self.handle_hotel_count),
            
            (r'special requests', self.handle_special_requests),
            (r'booking changes', self.handle_booking_changes)
        ]

        for pattern, handler in query_patterns:
            match = re.search(pattern, query)
            if match:
                # Check if the pattern includes a date capture group
                if match.groups():
                    original_answer = self.format_response(handler(match.group(1)))
                else:
                    original_answer = self.format_response(handler())
                
                # Optionally enhance with LLM
                if use_llm_enhancement:
                    return self.enhance_answer_with_llm(query, original_answer)
                
                return original_answer
        
        # Fallback to semantic search
        semantic_results = self.semantic_search(query)
        original_answer = self.format_response(self.format_semantic_search_results(semantic_results))
        
        return self.enhance_answer_with_llm(query, original_answer)

    
    def handle_monthly_revenue(self, month_year=None):
        # Handle revenue for a specific month or overall
        if month_year:
            # Try parsing the month and year
            try:
                # Handle different input formats
                if month_year.isdigit():
                    # If only year is provided
                    yearly_revenue = {}
                    for month, data in self.insights['revenue_by_month'].items():
                        if month.endswith(month_year):
                            yearly_revenue[month] = data
                    return yearly_revenue
                else:
                    # Full month and year (e.g., "July 2018")
                    return {month_year: self.insights['revenue_by_month'].get(month_year, "No data available")}
            except Exception as e:
                return f"Error parsing month/year: {str(e)}"
        
        # If no specific month, return entire monthly revenue
        return self.insights['revenue_by_month']
    
    def handle_cancellation_locations(self):
        # Return top locations with highest cancellations
        cancellation_by_location = self.insights['cancellation_by_location']
        
        # Return top 10 locations with most cancellations
        return {
            "Top 10 Locations with Highest Cancellations": 
            dict(list(cancellation_by_location.items())[:10])
        }
    
    def handle_hotel_count(self):
        # Directly return the count of different hotel types
        hotel_counts = self.df['hotel'].value_counts().to_dict()
        return f"Hotel Type Counts:\n" + "\n".join([f"{hotel_type}: {count}" for hotel_type, count in hotel_counts.items()])
    
    def format_response(self, data):
        # Convert Pandas Timestamp objects to string
        def convert_timestamps(obj):
            if isinstance(obj, pd.Timestamp):  # Check if it's a Timestamp
                return obj.isoformat()  # Convert to ISO format string
            return obj

        return json.dumps(data, indent=2, default=convert_timestamps)
    
    def handle_revenue_by_market_segment(self):
        return self.insights['revenue_by_market_segment']
    
    def handle_cancellation_summary(self):
        return self.insights['cancellation_summary']
    
    def handle_booking_trends(self):
        return self.insights['booking_trends']
    
    def handle_average_price(self):
        return f"Average Daily Rate (ADR): {self.df['adr'].mean():.2f}"
    
    def handle_special_requests(self):
        # Average number of special requests
        return f"Average Special Requests: {self.insights['hotel_stats']['avg_special_requests']:.2f}"
    
    def handle_booking_changes(self):
        # Average number of booking changes
        return f"Average Booking Changes: {self.insights['hotel_stats']['avg_booking_changes']:.2f}"
    
    def format_semantic_search_results(self, results):
        # Format search results into a readable format
        formatted_results = []
        for _, row in results.iterrows():
            formatted_results.append({
                'hotel_type': row.get('hotel', 'Unknown'),
                'country': row.get('country', 'Unknown'),
                'arrival_date': row.get('arrival_date', 'Unknown'),
                'market_segment': row.get('market_segment', 'Unknown'),
                'reservation_status': row.get('reservation_status', 'Unknown')
            })
        return formatted_results

def main():
    # Create RAG system with default or specified CSV path
    rag_system = HotelBookingRAG()
    
    # Example queries
    queries = [
        "Show me total revenue for July 2017.",
        "Which locations had the highest booking cancellations?",
        "how many resort hotel and city hotels are there?",
        "Show me total revenue for market segments",
        "What are the cancellation rates?",
        "Booking trends for different weeks",
        "What is the average price of a hotel booking?",
        "Find bookings similar to resort hotels in Portugal",
        "Tell me about special requests",
        "How many booking changes typically occur?",
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("Answer:")
        print(rag_system.answer_query(query))

if __name__ == '__main__':
    main()