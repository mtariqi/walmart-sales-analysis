# Create main.py with proper content
"""
Main entry point for Walmart Sales Analysis
"""

import argparse
import sys
from pathlib import Path

def main():
    """Main function to run the Walmart sales analysis pipeline."""
    parser = argparse.ArgumentParser(description="Walmart Sales Analysis Pipeline")
    parser.add_argument(
        "--data-path", 
        type=str, 
        default="data/raw",
        help="Path to raw data directory"
    )
    parser.add_argument(
        "--output-path", 
        type=str, 
        default="results",
        help="Path to output directory"
    )
    
    args = parser.parse_args()
    
    print("Walmart Sales Analysis Pipeline")
    print("===============================")
    print(f"Data path: {args.data_path}")
    print(f"Output path: {args.output_path}")
    print("\nTo run specific modules, import and use them directly.")
    print("Example:")
    print("  from src.data_loader import load_walmart_data")
    print("  from src.modeling import SalesForecaster")

if __name__ == "__main__":
    main()

