import sqlite3
import optuna
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import argparse



def analyze_optuna_results(db_path='sqlite:///optuna_autoencoder.db', study_name='autoencoder_opt'):
    """
    Analyze Optuna hyperparameter optimization results and visualize findings.
    
    Parameters:
    -----------
    db_path : str
        Path to SQLite database with Optuna results
    study_name : str
        Name of the Optuna study
    """
    # Load the study
    study = optuna.load_study(study_name=study_name, storage=db_path)
    
    # 1. Basic information
    print("=" * 50)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION RESULTS")
    print("=" * 50)
    print(f"Number of completed trials: {len(study.trials)}")
    print(f"Best trial number: {study.best_trial.number}")
    print(f"Best loss value: {study.best_value:.6f}")
    print("\nBest parameters:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")
    
    # 2. Create dataframe for analysis
    trials_df = study.trials_dataframe()
    
    # Check for additional metrics from user attributes
    has_evm = 'user_attrs_evm' in trials_df.columns
    
    # 3. Parameter distribution visualization
    plt.figure(figsize=(12, 10))
    
    # 3.1 Encoding dimension distribution
    plt.subplot(2, 2, 1)
    encoding_dims = sorted(trials_df['params_encoding_dim'].unique())
    encoding_counts = trials_df['params_encoding_dim'].value_counts().sort_index()
    plt.bar(encoding_counts.index, encoding_counts.values)
    plt.xlabel('Encoding Dimension')
    plt.ylabel('Count')
    plt.title('Distribution of Encoding Dimensions')
    
    # 3.2 Learning rate distribution
    plt.subplot(2, 2, 2)
    plt.hist(trials_df['params_learning_rate'], bins=20)
    plt.xlabel('Learning Rate')
    plt.ylabel('Count')
    plt.title('Distribution of Learning Rates')
    plt.xscale('log')
    
    # 3.3 Parameter impact on loss
    plt.subplot(2, 2, 3)
    encoding_dim_means = trials_df.groupby('params_encoding_dim')['value'].mean()
    encoding_dim_stds = trials_df.groupby('params_encoding_dim')['value'].std()
    x = np.arange(len(encoding_dim_means))
    plt.bar(x, encoding_dim_means, yerr=encoding_dim_stds, capsize=5)
    plt.xticks(x, encoding_dim_means.index)
    plt.xlabel('Encoding Dimension')
    plt.ylabel('Mean Loss (lower is better)')
    plt.title('Impact of Encoding Dimension on Loss')
    
    # 3.4 Learning rate vs loss
    plt.subplot(2, 2, 4)
    plt.scatter(trials_df['params_learning_rate'], trials_df['value'], alpha=0.6)
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate vs. Loss')
    plt.xscale('log')
    
    plt.tight_layout()
    plt.savefig('optuna_parameter_analysis.png')
    plt.show()
    
    # 4. Top 5 best trials
    print("\nTop 5 best trials:")
    top_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('inf'))[:5]
    for i, trial in enumerate(top_trials):
        print(f"Rank {i+1}:")
        print(f"  Trial number: {trial.number}")
        print(f"  Loss value: {trial.value:.6f}")
        print(f"  Parameters: {trial.params}")
        # Print additional metrics if available
        for key, value in trial.user_attrs.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    # 5. Parameter importance
    try:
        importances = optuna.importance.get_param_importances(study)
        print("\nParameter importance:")
        for param, importance in importances.items():
            print(f"  {param}: {importance:.4f}")
    except Exception as e:
        print(f"\nCould not calculate parameter importance: {e}")
        print("This usually requires completed trials with different parameter values")
    
    # 6. EVM analysis if available
    if has_evm:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        evm_by_dim = trials_df.groupby('params_encoding_dim')['user_attrs_evm_percent'].mean()
        plt.bar(evm_by_dim.index, evm_by_dim.values)
        plt.xlabel('Encoding Dimension')
        plt.ylabel('EVM (%)')
        plt.title('EVM vs Encoding Dimension')
        
        if 'user_attrs_compression_ratio' in trials_df.columns:
            plt.subplot(1, 2, 2)
            plt.scatter(trials_df['user_attrs_compression_ratio'], trials_df['user_attrs_evm_percent'], alpha=0.6)
            plt.xlabel('Compression Ratio')
            plt.ylabel('EVM (%)')
            plt.title('Compression Ratio vs EVM')
        
        plt.tight_layout()
        plt.savefig('optuna_evm_analysis.png')
        plt.show()
    
    # 7. Advanced visualizations (if visualization package is available)
    try:
        # Optimization history
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image('optuna_optimization_history.png')
        
        # Parallel coordinate plot
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_image('optuna_parallel_coordinate.png')
        
        # Parameter importance plot
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_image('optuna_param_importances.png')
        
        # Contour plot
        fig = optuna.visualization.plot_contour(study)
        fig.write_image('optuna_contour.png')
        
        print("\nVisualization images saved.")
    except Exception as e:
        print(f"\nCould not generate visualization images: {e}")
        print("Consider installing plotly for advanced visualizations.")
    
    return study, trials_df

if __name__ == "__main__":
    # Set up command line arguments
    # parser = argparse.ArgumentParser(description='Analyze Optuna hyperparameter optimization results')
    # parser.add_argument('--db-path', type=str, default='sqlite:///optuna_autoencoder.db',
    #                     help='Path to SQLite database with Optuna results')
    # parser.add_argument('--study-name', type=str, default='autoencoder_opt',
    #                     help='Name of the Optuna study')
    # parser.add_argument('--output-dir', type=str, default='.',
    #                     help='Directory to save analysis results')
    
    # args = parser.parse_args()
    
    # # Create output directory if it doesn't exist
    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir)
    
    # # Change to output directory
    # os.chdir(args.output_dir)
    
    study_name = "autoencoder_opt"
    db_path = "sqlite:///C:/Users/Aakarsh/Desktop/R&S-Hackathon/pca/optuna_autoencoder.db"


    # if not os.path.exists(db_path):
    #     raise FileNotFoundError(f"DB file not found at: {db_path}")
    # # Show raw contents (quick and dirty)
    # conn = sqlite3.connect(db_path)
    # cursor = conn.cursor()

    # # Check all study names
    # cursor.execute("SELECT study_name FROM studies;")
    # rows = cursor.fetchall()
    # print("Available studies in DB:")
    # for row in rows:
    #     print(row[0])

    # conn.close()
    # Run analysis
    study, trials_df = analyze_optuna_results(db_path=db_path, study_name=study_name)
    
    # Save dataframe to CSV
    output_file = f"{study_name}_analysis.csv"
    trials_df.to_csv(output_file, index=False)
    print(f"Analysis results saved to {output_file}")