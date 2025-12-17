"""
Usage:
    python compare_algorithms.py
    python compare_algorithms.py --algorithms hybrid sac td3
    python compare_algorithms.py --output comparison_report.pdf
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob


class AlgorithmComparator:
    """Compare results from multiple algorithms"""
    
    def __init__(self, algorithms=None):
        if algorithms is None:
            algorithms = ['hybrid', 'sac', 'td3', 'ppo']
        
        self.algorithms = algorithms
        self.results = {}
        
        print("\n" + "="*70)
        print("ALGORITHM COMPARISON TOOL")
        print("="*70)
        
    def load_results(self):
        """Load results from all algorithms"""
        print("\nLoading results...")
        
        for algo in self.algorithms:
            result_path = f'results/{algo}/objective3_performance_comparison.csv'
            
            if os.path.exists(result_path):
                df = pd.read_csv(result_path)
                self.results[algo] = df
                print(f"  ✓ Loaded {algo.upper()} results")
            else:
                print(f"  ✗ No results found for {algo.upper()}")
        
        if not self.results:
            print("\n⚠️  No results found! Run main.py first with different algorithms.")
            return False
        
        return True
    
    def compare_metrics(self):
        """Compare key metrics across algorithms"""
        print("\n" + "="*70)
        print("PERFORMANCE COMPARISON")
        print("="*70)
        
        # Key metrics to compare
        key_metrics = [
            'Horizontal Error (ft)',
            'Vertical Error (ft)',
            'Total 3D Error (ft)',
            'Max DLS (deg/100ft)',
            'Productivity Index'
        ]
        
        comparison_data = []
        
        for metric in key_metrics:
            row = {'Metric': metric}
            
            for algo in self.results:
                df = self.results[algo]
                metric_row = df[df['Metric'] == metric]
                
                if not metric_row.empty:
                    # Get the optimized value (first column after 'Metric')
                    col_name = [c for c in df.columns if 'Optimized' in c or algo.upper() in c][0]
                    value = metric_row[col_name].values[0]
                    row[algo.upper()] = value
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Print table
        print("\n" + comparison_df.to_string(index=False))
        
        # Find best algorithm for each metric
        print("\n" + "="*70)
        print("BEST PERFORMING ALGORITHM PER METRIC")
        print("="*70)
        
        for metric in key_metrics:
            row = comparison_df[comparison_df['Metric'] == metric].iloc[0]
            values = {algo: row[algo.upper()] for algo in self.results if algo.upper() in row}
            
            if 'Error' in metric or 'DLS' in metric:
                best_algo = min(values, key=values.get)
                best_value = values[best_algo]
            else:
                best_algo = max(values, key=values.get)
                best_value = values[best_algo]
            
            print(f"  {metric}: {best_algo.upper()} ({best_value:.2f})")
        
        return comparison_df
    
    def plot_comparison_bar(self, comparison_df, output_dir='comparison_plots'):
        """Create bar chart comparison"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nGenerating comparison plots in {output_dir}...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Algorithm Performance Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Distance Errors
        ax1 = axes[0, 0]
        error_metrics = ['Horizontal Error (ft)', 'Vertical Error (ft)', 'Total 3D Error (ft)']
        error_data = comparison_df[comparison_df['Metric'].isin(error_metrics)]
        
        x = np.arange(len(error_metrics))
        width = 0.2
        
        for i, algo in enumerate(self.results):
            if algo.upper() in error_data.columns:
                values = error_data[algo.upper()].values
                ax1.bar(x + i * width, values, width, label=algo.upper())
        
        ax1.set_ylabel('Error (ft)', fontsize=11)
        ax1.set_title('Distance Errors', fontsize=12, fontweight='bold')
        ax1.set_xticks(x + width * (len(self.results) - 1) / 2)
        ax1.set_xticklabels(['Horizontal', 'Vertical', 'Total 3D'], rotation=15)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: DLS Comparison
        ax2 = axes[0, 1]
        dls_metrics = ['Max DLS (deg/100ft)', 'Avg DLS (deg/100ft)']
        dls_data = comparison_df[comparison_df['Metric'].isin(dls_metrics)]
        
        x2 = np.arange(len(dls_metrics))
        
        for i, algo in enumerate(self.results):
            if algo.upper() in dls_data.columns:
                values = dls_data[algo.upper()].values
                ax2.bar(x2 + i * width, values, width, label=algo.upper())
        
        ax2.axhline(y=8, color='orange', linestyle='--', label='Target Limit (8°)')
        ax2.axhline(y=10, color='red', linestyle='--', label='Max Limit (10°)')
        ax2.set_ylabel('DLS (°/100ft)', fontsize=11)
        ax2.set_title('Dogleg Severity', fontsize=12, fontweight='bold')
        ax2.set_xticks(x2 + width * (len(self.results) - 1) / 2)
        ax2.set_xticklabels(['Max DLS', 'Avg DLS'])
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Productivity
        ax3 = axes[1, 0]
        prod_metric = comparison_df[comparison_df['Metric'] == 'Productivity Index']
        
        algos = [algo.upper() for algo in self.results if algo.upper() in prod_metric.columns]
        prod_values = [prod_metric[algo].values[0] for algo in algos]
        
        bars = ax3.bar(algos, prod_values, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'][:len(algos)])
        ax3.set_ylabel('Productivity Index', fontsize=11)
        ax3.set_title('Well Productivity', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}',
                    ha='center', va='bottom', fontsize=9)
        
        # Plot 4: Overall Score (lower error + higher productivity = better)
        ax4 = axes[1, 1]
        
        scores = {}
        for algo in self.results:
            if algo.upper() in comparison_df.columns:
                total_error = comparison_df[comparison_df['Metric'] == 'Total 3D Error (ft)'][algo.upper()].values[0]
                productivity = comparison_df[comparison_df['Metric'] == 'Productivity Index'][algo.upper()].values[0]
                
                # Normalize and compute score (lower is better for error, higher for productivity)
                # Score = normalized_productivity - normalized_error
                scores[algo.upper()] = productivity / 100000 - total_error / 1000
        
        algo_names = list(scores.keys())
        score_values = list(scores.values())
        
        colors = ['green' if s > 0 else 'red' for s in score_values]
        bars = ax4.barh(algo_names, score_values, color=colors, alpha=0.7)
        ax4.set_xlabel('Composite Score', fontsize=11)
        ax4.set_title('Overall Performance Score\n(Higher is Better)', fontsize=12, fontweight='bold')
        ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax4.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/algorithm_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved algorithm_comparison.png")
    
    def load_training_histories(self):
        """Load training history for all algorithms"""
        print("\nLoading training histories...")
        
        histories = {}
        
        for algo in self.algorithms:
            history_path = f'results/{algo}/objective2_training_history.csv'
            
            if os.path.exists(history_path):
                df = pd.read_csv(history_path)
                histories[algo] = df
                print(f"  ✓ Loaded {algo.upper()} training history")
        
        return histories
    
    def plot_training_comparison(self, histories, output_dir='comparison_plots'):
        if not histories:
            print("No training histories available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Training Progress Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Raw rewards
        for algo, df in histories.items():
            ax1.plot(df['Episode'], df['Reward'], 
                    label=algo.upper(), alpha=0.6, linewidth=1)
        
        ax1.set_xlabel('Episode', fontsize=11)
        ax1.set_ylabel('Episode Reward', fontsize=11)
        ax1.set_title('Training Rewards Over Time', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Smoothed rewards
        window = 20
        for algo, df in histories.items():
            if len(df) >= window:
                smoothed = df['Reward'].rolling(window=window, min_periods=1).mean()
                ax2.plot(df['Episode'], smoothed, 
                        label=algo.upper(), linewidth=2)
        
        ax2.set_xlabel('Episode', fontsize=11)
        ax2.set_ylabel('Smoothed Reward (20-ep window)', fontsize=11)
        ax2.set_title('Smoothed Training Progress', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/training_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved training_comparison.png")
    
    def generate_summary_report(self, comparison_df, output_file='comparison_report.txt'):
        print(f"\nGenerating summary report: {output_file}...")
        
        with open(output_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("WELL TRAJECTORY OPTIMIZATION - ALGORITHM COMPARISON REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write("ALGORITHMS COMPARED:\n")
            for algo in self.results:
                f.write(f"  - {algo.upper()}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("PERFORMANCE METRICS\n")
            f.write("="*70 + "\n\n")
            
            f.write(comparison_df.to_string(index=False))
            f.write("\n\n")
            
            # Best performers
            f.write("="*70 + "\n")
            f.write("BEST PERFORMING ALGORITHMS\n")
            f.write("="*70 + "\n\n")
            
            key_metrics = {
                'Total 3D Error (ft)': 'min',
                'Max DLS (deg/100ft)': 'min',
                'Productivity Index': 'max'
            }
            
            for metric, direction in key_metrics.items():
                row = comparison_df[comparison_df['Metric'] == metric].iloc[0]
                values = {algo: row[algo.upper()] for algo in self.results 
                         if algo.upper() in row}
                
                if direction == 'min':
                    best_algo = min(values, key=values.get)
                else:
                    best_algo = max(values, key=values.get)
                
                best_value = values[best_algo]
                f.write(f"{metric}:\n")
                f.write(f"  Winner: {best_algo.upper()} ({best_value:.2f})\n\n")
            
            # Recommendations
            f.write("="*70 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("="*70 + "\n\n")
            
            if 'hybrid' in self.results:
                f.write("HYBRID Algorithm:\n")
                f.write("  - Best for: Fast deployment, guaranteed feasibility\n")
                f.write("  - Pros: Quick training, reliable results\n")
                f.write("  - Use when: Time-constrained, production systems\n\n")
            
            if 'sac' in self.results:
                f.write("SAC Algorithm:\n")
                f.write("  - Best for: Optimal performance, exploration\n")
                f.write("  - Pros: State-of-the-art results, adaptive\n")
                f.write("  - Use when: Maximum optimization needed\n\n")
            
            if 'td3' in self.results:
                f.write("TD3 Algorithm:\n")
                f.write("  - Best for: Balance of performance and simplicity\n")
                f.write("  - Pros: Stable, good results\n")
                f.write("  - Use when: Good balance needed\n\n")
        
        print(f"  ✓ Saved {output_file}")
    
    def run_full_comparison(self):
        """Run complete comparison analysis"""
        # Load data
        if not self.load_results():
            return
        
        # Compare metrics
        comparison_df = self.compare_metrics()
        
        # Generate plots
        self.plot_comparison_bar(comparison_df)
        
        # Training comparison
        histories = self.load_training_histories()
        if histories:
            self.plot_training_comparison(histories)
        
        # Summary report
        self.generate_summary_report(comparison_df)
        
        print("\n" + "="*70)
        print("COMPARISON COMPLETE")
        print("="*70)
        print("\nOutputs:")
        print("  - comparison_plots/algorithm_comparison.png")
        print("  - comparison_plots/training_comparison.png")
        print("  - comparison_report.txt")
        print("\n✓ All comparisons completed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description='Compare results from different optimization algorithms'
    )
    parser.add_argument(
        '--algorithms',
        nargs='+',
        default=['hybrid', 'sac', 'td3', 'ppo'],
        help='Algorithms to compare'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='comparison_report.txt',
        help='Output report filename'
    )
    
    args = parser.parse_args()
    
    comparator = AlgorithmComparator(args.algorithms)
    comparator.run_full_comparison()


if __name__ == "__main__":
    main()