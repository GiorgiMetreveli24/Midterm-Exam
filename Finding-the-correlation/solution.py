import numpy as np
import matplotlib.pyplot as plt

def calculate_pearson_correlation():
    """
    Calculates Pearson's correlation coefficient for the given data
    and generates a scatter plot.
    """
    # Given data points
    x = np.array([-5, -3, -4, -1, 1, 3, 5, 7])
    y = np.array([2, -1, -4, 1, -2, 1, -3, -2])

    # Calculate Pearson's correlation coefficient
    correlation_matrix = np.corrcoef(x, y)
    pearson_correlation = correlation_matrix[0, 1]

    # Print the result
    print(f"Pearson's Correlation Coefficient: {pearson_correlation:.4f}")

    # Create a scatter plot for visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, label='Data Points')
   
    # Add a regression line
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m*x + b, color='red', label=f'Regression Line (r={pearson_correlation:.2f})')
   
    plt.title("Scatter Plot and Regression Line of the Given Data")
    plt.xlabel("X values")
    plt.ylabel("Y values")
    plt.grid(True)
    plt.legend()
   
    # Save the plot as an image
    plt.savefig("Finding-the-correlation/correlation_graph.png")
    print("Graph saved as correlation_graph.png")

if __name__ == '__main__':
    calculate_pearson_correlation()