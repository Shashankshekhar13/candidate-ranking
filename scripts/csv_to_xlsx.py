import sys
import pandas as pd

def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/csv_to_xlsx.py <input_csv> <output_xlsx>")
        sys.exit(1)
        
    input_csv = sys.argv[1]
    output_xlsx = sys.argv[2]
    
    try:
        df = pd.read_csv(input_csv)
        df.to_excel(output_xlsx, index=False)
        print(f"Successfully converted {input_csv} to {output_xlsx}")
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
