import pandas as pd
import matplotlib.pyplot as plt

def main():
    '''
    make plots out of the results
    plot 1: loss
    plot 2: grammaticality score
    plot 3: rankings of cn scores
    '''
    results_path = './training_results.csv'

    # Read the CSV file
    df = pd.read_csv(results_path)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: CEL
    ax1.bar(df['task_type'], df['CEL'], color='steelblue', edgecolor='black')
    ax1.set_xlabel('Task Type', fontsize=12, fontweight='bold')
    ax1.set_ylabel('CEL', fontsize=12, fontweight='bold')
    ax1.set_title('CEL by Task Type', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Plot 2: Perplexity
    ax2.bar(df['task_type'], df['perplexity'], color='coral', edgecolor='black')
    ax2.set_xlabel('Task Type', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Perplexity', fontsize=12, fontweight='bold')
    ax2.set_title('Perplexity by Task Type', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the figure
    plt.savefig('metrics_plots.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'metrics_plots.png'")

    # Display the plot
    plt.show()


if __name__ == "__main__":
    main()