import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    csv_path = 'healing_results.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    plt.figure(figsize=(10, 6))

    # Check that required columns exist
    methods_to_plot = {
        'gradient': 'Gradient Based',
        'regrowth': 'Neuron Regrowth',
        'rerouting': 'Dynamic Rerouting'
    }

    colors = ['blue', 'green', 'orange']
    markers = ['o', 's', '^']

    for i, (col, label) in enumerate(methods_to_plot.items()):
        if col in df.columns:
            plt.plot(df['Fault (%)'], df[col], marker=markers[i], color=colors[i], label=label, linewidth=2, markersize=8)
        else:
            print(f"Warning: Column '{col}' not found in data.")

    plt.title('Healing Performance Comparison for Selected Methods', fontsize=16)
    plt.xlabel('Fault (%)', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.xticks([10, 20, 30, 40])
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend to top right corner
    plt.legend(loc='upper right', fontsize=12)

    # Save the plot
    output_path = 'specific_healing_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to {output_path}")

if __name__ == '__main__':
    main()
