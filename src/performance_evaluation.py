import os
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from hotel_booking_rag import HotelBookingRAG

# Load environment variables from .env file
load_dotenv()

class RAGPerformanceEvaluator:
    def __init__(self, rag_system: HotelBookingRAG, llm_query_delay: float = 20):
        """
        Initialize performance evaluator for RAG system with query delay
        
        Args:
            rag_system (HotelBookingRAG): Initialized RAG system
            llm_query_delay (float): Delay in seconds between LLM-enhanced queries
        """
        self.rag_system = rag_system
        self.llm_query_delay = llm_query_delay
        self.last_llm_query_time = 0
        
        # Performance metrics storage
        self.performance_metrics = {
            'response_times': [],
            'query_results': [],
            'query_categories': {}
        }
    
    def generate_test_queries(self) -> List[Dict]:
        """
        Generate a comprehensive set of test queries with categories
        
        Returns:
            List[Dict]: Comprehensive test queries with metadata
        """
        return [
            # Revenue Queries
            {"query": "Total revenue for July 2017", "category": "revenue"},
            {"query": "Revenue by market segment", "category": "revenue"},
            
            # Cancellation Queries
            {"query": "Cancellation rates", "category": "cancellation"},
            {"query": "Locations with highest cancellations", "category": "cancellation"},
            
            # Booking Trends
            {"query": "Booking trends for different weeks", "category": "trends"},
            {"query": "Average lead time for bookings", "category": "trends"},
            
            # Hotel Specifics
            {"query": "How many resort and city hotels are there?", "category": "hotel_count"},
            {"query": "Average price of hotel bookings", "category": "pricing"},
            
            # Special Metrics
            {"query": "Average special requests", "category": "special_metrics"},
            {"query": "Average booking changes", "category": "special_metrics"},
            
            # Semantic Search Queries
            {"query": "Find bookings similar to resort hotels in Portugal", "category": "semantic_search"},
            {"query": "Bookings with unique characteristics", "category": "semantic_search"}
        ]
    
    def evaluate_query_performance(self, query_info: Dict, use_llm_enhancement: bool = True) -> Dict:
        """
        Comprehensive performance evaluation for a single query with delay mechanism
        
        Args:
            query_info (Dict): Query information
            use_llm_enhancement (bool): Whether to use LLM enhancement
        
        Returns:
            Dict: Detailed performance metrics for the query
        """
        # Implement delay for LLM-enhanced queries
        if use_llm_enhancement:
            current_time = time.time()
            time_since_last_query = current_time - self.last_llm_query_time
            
            # If not enough time has passed since the last LLM query, wait
            if time_since_last_query < self.llm_query_delay:
                wait_time = self.llm_query_delay - time_since_last_query
                print(f"Waiting {wait_time:.2f} seconds before LLM query to manage rate limits")
                time.sleep(wait_time)
            
            # Update the last LLM query time
            self.last_llm_query_time = time.time()
        
        start_time = time.time()
        
        try:
            # Perform query
            response = self.rag_system.answer_query(
                query_info['query'], 
                use_llm_enhancement=use_llm_enhancement
            )
            
            end_time = time.time()
            
            # Calculate metrics
            query_metrics = {
                'query': query_info['query'],
                'category': query_info['category'],
                'response_time_ms': (end_time - start_time) * 1000,
                'response_length': len(response),
                'use_llm_enhancement': use_llm_enhancement,
                'response': response
            }
            
            # Track performance by category
            if query_info['category'] not in self.performance_metrics['query_categories']:
                self.performance_metrics['query_categories'][query_info['category']] = []
            
            self.performance_metrics['query_categories'][query_info['category']].append(query_metrics)
            self.performance_metrics['response_times'].append(query_metrics)
            self.performance_metrics['query_results'].append(query_metrics)
            
            return query_metrics
        
        except Exception as e:
            return {
                'query': query_info['query'],
                'category': query_info['category'],
                'error': str(e),
                'response_time_ms': None
            }
    
    def compute_performance_summary(self) -> Dict:
        """
        Compute comprehensive performance summary
        
        Returns:
            Dict: Detailed performance analysis
        """
        # Filter out queries with errors
        response_times = [
            metric['response_time_ms'] 
            for metric in self.performance_metrics['response_times'] 
            if metric['response_time_ms'] is not None
        ]
        
        # Category-wise performance
        category_performance = {}
        for category, metrics in self.performance_metrics['query_categories'].items():
            category_times = [m['response_time_ms'] for m in metrics if m.get('response_time_ms')]
            category_performance[category] = {
                'total_queries': len(metrics),
                'avg_response_time_ms': np.mean(category_times) if category_times else None,
                'max_response_time_ms': np.max(category_times) if category_times else None,
                'min_response_time_ms': np.min(category_times) if category_times else None
            }
        
        return {
            'total_queries': len(self.performance_metrics['response_times']),
            'avg_response_time_ms': np.mean(response_times) if response_times else None,
            'median_response_time_ms': np.median(response_times) if response_times else None,
            'max_response_time_ms': np.max(response_times) if response_times else None,
            'min_response_time_ms': np.min(response_times) if response_times else None,
            'queries_with_llm_enhancement': sum(
                1 for metric in self.performance_metrics['response_times'] 
                if metric.get('use_llm_enhancement', False)
            ),
            'category_performance': category_performance
        }
    
    def generate_performance_visualizations(self):
        """
        Generate performance visualization charts
        """
        # Response Time by Category
        plt.figure(figsize=(12, 6))
        category_response_times = {}
        for category, metrics in self.performance_metrics['query_categories'].items():
            category_response_times[category] = [
                m['response_time_ms'] for m in metrics if m.get('response_time_ms')
            ]
        
        plt.boxplot(
            [times for times in category_response_times.values()], 
            labels=list(category_response_times.keys())
        )
        plt.title('Query Response Times by Category')
        plt.xlabel('Query Category')
        plt.ylabel('Response Time (ms)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('response_times_by_category.png')
        plt.close()
        
        # LLM Enhancement Impact
        llm_times = [
            m['response_time_ms'] for m in self.performance_metrics['response_times'] 
            if m.get('use_llm_enhancement') and m.get('response_time_ms')
        ]
        non_llm_times = [
            m['response_time_ms'] for m in self.performance_metrics['response_times'] 
            if not m.get('use_llm_enhancement') and m.get('response_time_ms')
        ]
        
        plt.figure(figsize=(10, 6))
        plt.boxplot([non_llm_times, llm_times], labels=['Without LLM', 'With LLM'])
        plt.title('Impact of LLM Enhancement on Response Times')
        plt.ylabel('Response Time (ms)')
        plt.tight_layout()
        plt.savefig('llm_enhancement_impact.png')
        plt.close()
    
    def generate_performance_report(self, output_path: str = 'performance_report.json'):
        """
        Generate a comprehensive performance report
        
        Args:
            output_path (str): Path to save performance report
        """
        # Compute performance summary
        performance_summary = self.compute_performance_summary()
        
        # Prepare detailed report
        report = {
            'performance_summary': performance_summary,
            'detailed_query_metrics': self.performance_metrics['query_results']
        }
        
        # Save report to JSON
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate visualizations
        self.generate_performance_visualizations()
        
        print(f"Performance report saved to {output_path}")
        print("Visualization charts saved: response_times_by_category.png, llm_enhancement_impact.png")
    
    def run_comprehensive_evaluation(self):
        """
        Run comprehensive performance evaluation with delay between LLM queries
        """
        # Generate test queries
        test_queries = self.generate_test_queries()
        
        # Evaluate queries with and without LLM enhancement
        for query_info in test_queries:
            # Without LLM enhancement
            self.evaluate_query_performance(query_info, use_llm_enhancement=False)
            
            # With LLM enhancement
            self.evaluate_query_performance(query_info, use_llm_enhancement=True)
        
        # Generate performance report
        self.generate_performance_report()

def main():
    # Get CSV path from environment variable or use a default
    csv_path = os.getenv('HOTEL_BOOKINGS_CSV_PATH', 'data/cleaned_hotel_bookings.csv')
    
    # Get Gemini API key from environment variable
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file. Please set it.")
    
    # Update this part based on the actual initialization of HotelBookingRAG
    # This is a placeholder - you'll need to adjust it to match the actual constructor
    rag_system = HotelBookingRAG(
        csv_path=csv_path
    )
    
    # If the RAG system needs the API key set separately, do it like this
    # Uncomment and modify as needed
    # rag_system.set_api_key(gemini_api_key)
    
    # Create performance evaluator with 20-second delay between LLM queries
    evaluator = RAGPerformanceEvaluator(rag_system, llm_query_delay=20)
    
    # Run comprehensive evaluation
    evaluator.run_comprehensive_evaluation()

if __name__ == '__main__':
    main()