import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class HotelBookingAnalytics:
    def __init__(self, data_path=None):
        """
        Initialize analytics with hotel bookings data
        
        Args:
            data_path (str, optional): Path to the cleaned CSV file. 
                                       If None, attempts to find the file in default locations.
        """
        # Define potential default file paths
        default_paths = [
            'data/processed/cleaned_hotel_bookings.csv',
            'data/cleaned_hotel_bookings.csv',
            '../data/processed/cleaned_hotel_bookings.csv',
            'cleaned_hotel_bookings.csv',
            os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'cleaned_hotel_bookings.csv')
        ]
        
        # Create output directory for visualizations
        os.makedirs('analytics_output', exist_ok=True)
        
        # Find the first existing path
        if data_path is None:
            for path in default_paths:
                if os.path.exists(path):
                    data_path = path
                    break
            else:
                raise FileNotFoundError("Could not find cleaned hotel bookings CSV. "
                                        "Please provide the correct path or ensure the file exists.")
        
        # Verify the file exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"The specified data path does not exist: {data_path}")
        
        # Read and preprocess data
        self.df = pd.read_csv(data_path)
        self.df['arrival_date'] = pd.to_datetime(self.df['arrival_date'])
        
        # Ensure required columns are present
        if 'is_canceled' not in self.df.columns:
            self.df['is_canceled'] = 0
        if 'total_nights' not in self.df.columns:
            self.df['total_nights'] = self.df['stays_in_weekend_nights'] + self.df['stays_in_week_nights']
        
    def revenue_trends(self):
        """
        Calculate and visualize monthly revenue trends
        
        Returns:
            dict: Monthly revenue insights
        """
        # Group by month and calculate total revenue
        monthly_revenue = self.df.groupby(
            pd.Grouper(key='arrival_date', freq='M')
        )['adr'].sum().reset_index()
        
        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(monthly_revenue['arrival_date'], monthly_revenue['adr'], marker='o')
        plt.title('Monthly Revenue Trends')
        plt.xlabel('Date')
        plt.ylabel('Total Revenue')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('analytics_output/revenue_trends.png')
        plt.close()

        return {
            'monthly_revenue': monthly_revenue.to_dict(orient='records'),
            'total_revenue': self.df['adr'].sum(),
            'average_monthly_revenue': monthly_revenue['adr'].mean()
        }

    def cancellation_analysis(self):
        """
        Analyze booking cancellations and related factors
        
        Returns:
            dict: Cancellation insights
        """
        total_bookings = len(self.df)
        canceled_bookings = self.df['is_canceled'].sum()
        cancellation_rate = (canceled_bookings / total_bookings) * 100

        # Cancellation correlation analysis
        # Calculate lead time if not already present
        if 'lead_time' not in self.df.columns:
            self.df['lead_time'] = (self.df['arrival_date'] - pd.Timestamp.now()).dt.days

        cancellation_factors = {
            'lead_time_correlation': np.corrcoef(self.df['lead_time'], self.df['is_canceled'])[0, 1],
            'adr_correlation': np.corrcoef(self.df['adr'], self.df['is_canceled'])[0, 1]
        }

        # Cancellation by hotel and room type
        cancellation_breakdown = {
            'by_hotel_type': self.df.groupby('hotel')['is_canceled'].mean() * 100,
            'by_room_type': self.df.groupby('reserved_room_type')['is_canceled'].mean() * 100
        }

        # Cancellation by hotel type
        cancellation_by_hotel = self.df.groupby('hotel')['is_canceled'].mean() * 100

        # Visualization of cancellation factors
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        sns.boxplot(x='is_canceled', y='lead_time', data=self.df)
        plt.title('Lead Time vs Cancellation')
        
        plt.subplot(1, 3, 2)
        sns.boxplot(x='is_canceled', y='adr', data=self.df)
        plt.title('Average Daily Rate vs Cancellation')

        plt.subplot(1, 3, 3)
        cancellation_by_hotel.plot(kind='bar')
        plt.title('Cancellation Rate by Hotel Type')
        plt.xlabel('Hotel Type')
        plt.ylabel('Cancellation Rate (%)')
        
        plt.tight_layout()
        plt.savefig('analytics_output/cancellation_factors.png')
        plt.close()

        return {
            'total_bookings': total_bookings,
            'canceled_bookings': canceled_bookings,
            'cancellation_rate': cancellation_rate,
            'cancellation_factors': cancellation_factors,
            'cancellation_breakdown': {
                k: v.to_dict() for k, v in cancellation_breakdown.items()
            },
            'cancellation_by_hotel': cancellation_by_hotel.to_dict()
        }

    def geographical_distribution(self):
        """
        Analyze geographical distribution of bookings
        
        Returns:
            dict: Booking distribution by country
        """
        # Count bookings by country
        country_bookings = self.df['country'].value_counts()
        
        # Top 10 countries
        top_countries = country_bookings.head(10)

        # Plotting
        plt.figure(figsize=(12, 6))
        top_countries.plot(kind='bar')
        plt.title('Top 10 Countries by Number of Bookings')
        plt.xlabel('Country')
        plt.ylabel('Number of Bookings')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('analytics_output/geographical_distribution.png')
        plt.close()

        return {
            'total_countries': len(country_bookings),
            'top_10_countries': top_countries.to_dict()
        }

    def booking_patterns(self):
        """
        Analyze booking patterns and seasonality
        
        Returns:
            dict: Booking pattern insights
        """
        # Extract month and day of week
        self.df['month'] = self.df['arrival_date'].dt.month
        self.df['day_of_week'] = self.df['arrival_date'].dt.day_name()

        # Create heatmap of bookings
        bookings_heatmap = self.df.groupby(['month', 'day_of_week']).size().unstack()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(bookings_heatmap, cmap='YlGnBu', annot=True, fmt='g')
        plt.title('Booking Heatmap: Month vs Day of Week')
        plt.tight_layout()
        plt.savefig('analytics_output/booking_heatmap.png')
        plt.close()

        # Seasonality analysis
        seasonal_revenue = self.df.groupby(
            pd.cut(self.df['arrival_date'].dt.month, 
                   bins=[0, 3, 6, 9, 12], 
                   labels=['Winter', 'Spring', 'Summer', 'Fall'])
        )['adr'].agg(['mean', 'count'])

        return {
            'bookings_heatmap': bookings_heatmap.to_dict(),
            'seasonal_revenue': seasonal_revenue.to_dict()
        }

    def booking_lead_time(self):
        """
        Analyze booking lead time distribution
        
        Returns:
            dict: Lead time insights
        """
        # Calculate lead time
        self.df['lead_time'] = (self.df['arrival_date'] - pd.Timestamp.now()).dt.days

        # Distribution plot
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df['lead_time'], kde=True)
        plt.title('Booking Lead Time Distribution')
        plt.xlabel('Lead Time (Days)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig('analytics_output/lead_time_distribution.png')
        plt.close()

        return {
            'average_lead_time': self.df['lead_time'].mean(),
            'median_lead_time': self.df['lead_time'].median(),
            'min_lead_time': self.df['lead_time'].min(),
            'max_lead_time': self.df['lead_time'].max()
        }

    def customer_segmentation(self):
        """
        Perform customer segmentation
        
        Returns:
            dict: Customer segmentation insights
        """
        # Prepare features for clustering
        features = ['total_nights', 'adr', 'adults', 'children']
        
        # Prepare data for clustering
        clustering_data = self.df[features].copy()
        
        # Normalize features
        scaler = StandardScaler()
        clustering_scaled = scaler.fit_transform(clustering_data)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        self.df['customer_segment'] = kmeans.fit_predict(clustering_scaled)

        # Segment characteristics
        segment_profiles = self.df.groupby('customer_segment')[features].mean()

        # Visualization - Violin Plot
        plt.figure(figsize=(12, 22))
        for i, feature in enumerate(features, 1):
            plt.subplot(4, 1, i)
            sns.violinplot(x='customer_segment', y=feature, data=self.df, inner="quartile", palette="pastel")
            plt.title(f'{feature} by Customer Segment')
        plt.tight_layout()
        plt.savefig('analytics_output/customer_segments.png')
        plt.close()

        return {
            'segment_profiles': segment_profiles.to_dict(),
            'segment_distribution': self.df['customer_segment'].value_counts().to_dict()
        }

    def stay_duration_impact(self):
        """
        Analyze impact of stay duration on revenue
        
        Returns:
            dict: Stay duration insights
        """
        # Calculate total stay duration if not already present
        if 'total_nights' not in self.df.columns:
            self.df['total_nights'] = self.df['stays_in_weekend_nights'] + self.df['stays_in_week_nights']
        
        total_nights = self.df['total_nights']
        
        # Group by total nights and calculate average revenue
        stay_duration_revenue = self.df.groupby(total_nights)['adr'].mean()
        
        # Visualization
        plt.figure(figsize=(12, 6))
        stay_duration_revenue.plot(kind='bar')
        plt.title('Average Daily Rate by Stay Duration')
        plt.xlabel('Total Nights')
        plt.ylabel('Average Daily Rate')
        plt.tight_layout()
        plt.savefig('analytics_output/stay_duration_revenue.png')
        plt.close()

        return {
            'avg_revenue_by_nights': stay_duration_revenue.to_dict(),
            'correlation_nights_revenue': np.corrcoef(total_nights, self.df['adr'])[0, 1]
        }

    def generate_comprehensive_report(self):
        """
        Generate a comprehensive analytics report
        
        Returns:
            dict: Comprehensive analytics insights
        """
        return {
            'revenue_trends': self.revenue_trends(),
            'cancellation_analysis': self.cancellation_analysis(),
            'geographical_distribution': self.geographical_distribution(),
            'booking_lead_time': self.booking_lead_time(),
            'booking_patterns': self.booking_patterns(),
            'customer_segmentation': self.customer_segmentation(),
            'stay_duration_impact': self.stay_duration_impact()
        }

# Run analytics if script is executed directly
def convert_to_serializable(obj):
    """
    Convert non-JSON serializable objects to serializable formats
    """
    import pandas as pd
    import numpy as np
    
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

if __name__ == '__main__':
    # Example usage with optional path
    analytics = HotelBookingAnalytics()  # Will auto-detect data path
    # Or specify a custom path
    # analytics = HotelBookingAnalytics('path/to/your/cleaned_hotel_bookings.csv')
    
    report = analytics.generate_comprehensive_report()
    
    # Print the report in a more readable format
    import json
    
    # Use a custom default function to handle non-serializable types
    print(json.dumps(report, indent=2, default=convert_to_serializable))