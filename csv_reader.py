import os
import glob
import pandas as pd


def load_and_combine_csvs(csv_directory, pattern="*.csv"):
    """
    Loads all CSV files from a directory that match the given pattern,
    parses the 'timestamp' column as datetime, and combines them into one DataFrame.

    Args:
        csv_directory (str): Path to the directory containing the CSV files.
        pattern (str): File pattern to match CSV files. Default is "*.csv".

    Returns:
        pd.DataFrame: A combined DataFrame containing data from all CSV files.
    """
    # Find all CSV files in the specified directory that match the pattern
    csv_files = glob.glob(os.path.join(csv_directory, pattern))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {csv_directory} matching pattern {pattern}")

    # List to hold DataFrames from each CSV
    dfs = []
    for file in csv_files:
        print(f"Loading file: {file}")
        try:
            # Read each CSV file, parsing the 'Timestamp' column as datetime
            df = pd.read_csv(file, parse_dates=["Timestamp"])
        except ValueError as e:
            print(f"Error parsing 'Timestamp' in file {file}: {e}")
            continue  # Skip this file and continue with others

        # Check if 'Timestamp' column exists
        if 'Timestamp' not in df.columns:
            print(f"'Timestamp' column not found in file {file}. Skipping this file.")
            continue

        # Step 1: Round 'Timestamp' to the nearest minute
        df['Timestamp'] = df['Timestamp'].dt.round('min')  # 'min' stands for minute frequency

        # Step 2: Filter out timestamps before 6:00 AM and after 6:00 PM
        df = df[(df['Timestamp'].dt.hour >= 6) & (df['Timestamp'].dt.hour <= 18)]

        # Append the processed DataFrame to the list
        dfs.append(df)

    if not dfs:
        raise ValueError("No data to combine after processing all CSV files.")

    # Concatenate all DataFrames into one combined DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)

    # Step 3: Remove duplicate timestamps, keeping the first occurrence
    combined_df = combined_df.drop_duplicates(subset=['Timestamp'], keep='first')

    combined_df = combined_df.reset_index(drop=True)

    return combined_df


# Example usage:
if __name__ == "__main__":
    # Directory where your CSV files are stored
    csv_directory = r"C:\Users\jldag\OneDrive\Desktop\Senior-Design\solar_radiation data" #file path for training csv

    # Load and combine the CSV files
    combined_df = load_and_combine_csvs(csv_directory)

    # Print some information about the combined DataFrame
    print("Combined DataFrame shape:", combined_df.shape)
    print("First few rows:")
    print(combined_df.head(500))