#!/usr/bin/env python3
"""
Data Processing Utility for Retail Sales Forecast Assistant
"""

import pandas as pd
import numpy as np
import os
import sys

def reduce_dataset(input_file, output_file, sample_ratio=0.2, random_state=42):
    """
    Reduce dataset size by randomly sampling records without bias
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
        sample_ratio (float): Ratio of data to keep (default: 0.2 = 20%)
        random_state (int): Random seed for reproducibility
    """
    try:
        print(f"Reading dataset from {input_file}...")
        df = pd.read_csv(input_file, encoding='latin-1')
        print(f"Original dataset shape: {df.shape}")
        
        # Randomly sample data
        sample_size = int(len(df) * sample_ratio)
        reduced_df = df.sample(n=sample_size, random_state=random_state)
        
        print(f"Reduced dataset shape: {reduced_df.shape}")
        
        # Save the reduced dataset
        reduced_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Reduced dataset saved as '{output_file}'")
        
        # Display statistics
        print(f"\nDataset size reduced from {len(df):,} to {len(reduced_df):,} records")
        print(f"Memory usage reduced from {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB to {reduced_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"Error processing dataset: {str(e)}")
        return False

def validate_data_format(file_path):
    """
    Validate that the CSV file has the required columns
    
    Args:
        file_path (str): Path to CSV file
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        df = pd.read_csv(file_path, encoding='latin-1')
        
        required_columns = ['Order_Date', 'Sales']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return False
        
        # Check data types
        try:
            pd.to_datetime(df['Order_Date'])
            pd.to_numeric(df['Sales'])
        except Exception as e:
            print(f"❌ Data type validation failed: {str(e)}")
            return False
        
        print(f"✅ Data format validation passed!")
        print(f"   - Shape: {df.shape}")
        print(f"   - Date range: {df['Order_Date'].min()} to {df['Order_Date'].max()}")
        print(f"   - Sales range: ${df['Sales'].min():.2f} - ${df['Sales'].max():.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error validating data: {str(e)}")
        return False

def main():
    """Main function for data processing"""
    if len(sys.argv) < 2:
        print("Usage: python data_processor.py <command> [options]")
        print("Commands:")
        print("  reduce <input_file> <output_file> [sample_ratio]")
        print("  validate <file_path>")
        return
    
    command = sys.argv[1]
    
    if command == "reduce":
        if len(sys.argv) < 4:
            print("Usage: python data_processor.py reduce <input_file> <output_file> [sample_ratio]")
            return
        
        input_file = sys.argv[2]
        output_file = sys.argv[3]
        sample_ratio = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2
        
        if not os.path.exists(input_file):
            print(f"❌ Input file not found: {input_file}")
            return
        
        reduce_dataset(input_file, output_file, sample_ratio)
        
    elif command == "validate":
        if len(sys.argv) < 3:
            print("Usage: python data_processor.py validate <file_path>")
            return
        
        file_path = sys.argv[2]
        
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return
        
        validate_data_format(file_path)
        
    else:
        print(f"❌ Unknown command: {command}")

if __name__ == "__main__":
    main()
