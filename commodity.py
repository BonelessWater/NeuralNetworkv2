# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to load and prepare the data
def load_data(file_path):
    """
    This function loads the dataset from a CSV file.
    Each column should represent a different commodity, metal, or industrial material.
    """
    data = pd.read_csv(file_path)
    return data

# Function to calculate correlation
def calculate_correlation(data):
    """
    This function calculates the correlation matrix for the given dataset.
    """
    correlation_matrix = data.corr()
    return correlation_matrix

# Function to visualize the correlation using matplotlib
def visualize_correlation(correlation_matrix):
    """
    This function visualizes the correlation matrix using matplotlib.
    """
    plt.figure(figsize=(12, 8))
    plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    
    # Add labels and title
    plt.xticks(np.arange(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
    plt.yticks(np.arange(len(correlation_matrix.columns)), correlation_matrix.columns)
    plt.title('Correlation Matrix of Commodities, Metals, and Industry Materials')
    
    # Add correlation values on the matrix
    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.columns)):
            plt.text(j, i, round(correlation_matrix.iloc[i, j], 2),
                     ha="center", va="center", color="black")
    
    plt.tight_layout()
    plt.show()

# Main function to execute the program
def main():
    # Load your dataset
    file_path = 'commodity_data.csv'  # Replace this with the path to your data file
    data = load_data(file_path)
    
    # Calculate correlation
    correlation_matrix = calculate_correlation(data)
    
    # Visualize correlation
    visualize_correlation(correlation_matrix)

if __name__ == "__main__":
    main()
