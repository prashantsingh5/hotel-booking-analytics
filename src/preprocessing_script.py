import pandas as pd
import numpy as np
import os
import logging

def preprocess_hotel_bookings(input_file='data/raw/hotel_bookings.csv'):
    """
    Comprehensive preprocessing function for hotel bookings dataset
    
    Args:
        input_file (str): Path to the input CSV file
    
    Returns:
        pd.DataFrame: Cleaned and preprocessed DataFrame
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)

    try:
        # 1. Load dataset
        df = pd.read_csv(input_file)
        logger.info(f"Original dataset shape: {df.shape}")

        # 2. Drop duplicates
        df.drop_duplicates(inplace=True)
        logger.info(f"Dropped duplicates. New shape: {df.shape}")

        # 3. Advanced missing value handling
        # More nuanced approach to filling missing values
        df['children'] = df['children'].fillna(0)
        df['agent'] = df['agent'].fillna(0)
        df['company'] = df['company'].fillna(0)
        df['country'] = df['country'].fillna('Unknown')

        # 4. Create a unified arrival_date column
        df['arrival_date'] = pd.to_datetime(
            df['arrival_date_year'].astype(str) + '-' +
            df['arrival_date_month'] + '-' +
            df['arrival_date_day_of_month'].astype(str),
            errors='coerce'
        )

        # 5. Convert reservation_status_date to datetime
        df['reservation_status_date'] = pd.to_datetime(
            df['reservation_status_date'], 
            errors='coerce', 
            dayfirst=True
        )

        # 6. Advanced data quality checks
        # Remove rows with impossible or extreme values
        df = df[
            (df['adr'] >= 0) &  # Non-negative average daily rate
            (df['stays_in_weekend_nights'] >= 0) &
            (df['stays_in_week_nights'] >= 0) &
            (df['adults'] >= 0) &
            (df['children'] >= 0)
        ]

        # 7. Feature engineering
        # Total stay duration
        df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
        
        # Booking status binary encoding
        df['is_canceled'] = df['is_canceled'].astype(int)
        
        # 8. Remove unnecessary columns
        columns_to_drop = [
            'arrival_date_year', 
            'arrival_date_month', 
            'arrival_date_day_of_month',
            'stays_in_weekend_nights',
            'stays_in_week_nights'
        ]
        df.drop(columns=columns_to_drop, axis=1, inplace=True)

        # 9. Reset index
        df.reset_index(drop=True, inplace=True)

        logger.info(f"Final preprocessed dataset shape: {df.shape}")

        # 10. Create output directory
        os.makedirs('data', exist_ok=True)
        
        # 11. Save preprocessed data
        output_file = 'data/processed/cleaned_hotel_bookings.csv'
        df.to_csv(output_file, index=False)
        logger.info(f"Preprocessed data saved to {output_file}")

        return df

    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise

# Run preprocessing if script is executed directly
if __name__ == '__main__':
    preprocessed_df = preprocess_hotel_bookings()