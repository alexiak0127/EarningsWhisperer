# visualizations.py - Creates visualizations for project results

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

# Create visualizations directory
os.makedirs('visualizations', exist_ok=True)

# plot style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# -- load results from results/ --
def load_results():

    results = {}
    
    # Load model comparison CSV
    comparison_file = 'results/model_comparison.csv'
    if os.path.exists(comparison_file):
        results['comparison'] = pd.read_csv(comparison_file)
        print(f"Loaded model comparison from {comparison_file}")
    
    # Load all confusion matrices
    confusion_files = glob.glob('results/*_confusion.csv')
    results['confusion'] = {}
    for file in confusion_files:
        model_name = os.path.basename(file).split('_confusion')[0]
        results['confusion'][model_name] = pd.read_csv(file, index_col=0)
        print(f"Loaded confusion matrix for {model_name}")
    
    # load all coefficient files
    coef_files = glob.glob('results/logistic_regression_coefficients_*.csv')
    if coef_files:
        results['coefficients'] = {}
        for file in coef_files:
            class_name = os.path.basename(file).split('_')[-1].split('.')[0]
            results['coefficients'][class_name] = pd.read_csv(file)
            print(f"Loaded coefficients for class {class_name}")
    
    # load predictions
    prediction_files = glob.glob('results/*_predictions.csv')
    results['predictions'] = {}
    for file in prediction_files:
        model_name = os.path.basename(file).split('_predictions')[0]
        results['predictions'][model_name] = pd.read_csv(file)
        print(f"Loaded predictions for {model_name}")
    
    return results


# -- generate model comparison bar chart --

def visualize_model_comparison(comparison_df):
    plt.figure(figsize=(8, 6))
    
    # Create bar chart
    ax = sns.barplot(x='Model', y='Accuracy', data=comparison_df)
    
    # Add value labels
    for i, row in comparison_df.iterrows():
        ax.text(i, row['Accuracy'] + 0.01, f"{row['Accuracy']:.4f}", 
                ha='center', va='bottom', fontsize=11)
    
    plt.title('Model Accuracy Comparison', fontsize=14)
    plt.ylim(0, min(1.0, comparison_df['Accuracy'].max() + 0.1))
    plt.ylabel('Accuracy', fontsize=12)
    plt.xlabel('', fontsize=12)
    plt.xticks(fontsize=11)
    plt.tight_layout()
    
    # Save figure
    plt.savefig('visualizations/model_comparison.png', dpi=300)
    plt.close()
    print("Created model comparison visualization")


# --  generate Confusion Matrix Heatmaps --

def visualize_confusion_matrices(confusion_dict):

    for model_name, cm_df in confusion_dict.items():
        plt.figure(figsize=(7, 6))
        
        # extract the values + numeric
        cm = cm_df.values.astype(float)
        
        # normalize each row
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        
        # replace division errors with 0
        cm_norm = np.nan_to_num(cm_norm)
        
        # Plot confusion matrix
        matrix = sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
                    xticklabels=cm_df.columns, yticklabels=cm_df.index)
        
        model_title = ' '.join(word.capitalize() for word in model_name.split('_'))
        plt.title(f'{model_title} Confusion Matrix', fontsize=14)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(f'visualizations/{model_name}_confusion_matrix.png', dpi=300)
        plt.close()
        
        print(f"Created confusion matrix visualization for {model_name}")


# -- Generate Logistic Regression Coefficients --

def visualize_lr_coefficients(coefficients_dict):
    
    # check - fixed
    if not coefficients_dict:
        print("No coefficient data available")
        return
    
    for class_name, coef_df in coefficients_dict.items():
        plt.figure(figsize=(10, 6))
        
        # Sort from highest to lowest
        coef_df = coef_df.sort_values('Coefficient', ascending=False)
        
        # plot
        sns.barplot(x='Coefficient', y='Feature', data=coef_df)
        
        # Add a line at 0 to separate positive vs negative coefficients
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        
        class_title = class_name.capitalize()
        plt.title(f'Logistic Regression Coefficients - {class_title} Class', fontsize=14)
        plt.xlabel('Coefficient Value', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(f'visualizations/lr_coefficients_{class_name}.png', dpi=300)
        plt.close()
        
        print(f"Created coefficient visualization for {class_name} class")


#  -- Generate Prediction Accuracy by Class --

def create_prediction_analysis(predictions_dict):
   
    # check - fixed
    if not predictions_dict:
        print("No prediction data available")
        return
    
    # Combine predictions from all models
    all_predictions = pd.DataFrame()
    for model_name, pred_df in predictions_dict.items():
        pred_df['model'] = model_name
        all_predictions = pd.concat([all_predictions, pred_df])
    
    # Create a heatmap of where models agree/disagree
    plt.figure(figsize=(8, 6))
    
    # Add a "correct" column to show whether the prediction was right
    all_predictions['correct'] = all_predictions['pred'] == all_predictions['true']

    # Make a summary table
    success_by_class = all_predictions.pivot_table(
        index='true',
        columns='model',
        values='correct',
        aggfunc='mean'
    )   
    
    sns.heatmap(success_by_class, annot=True, cmap='YlGnBu', fmt='.2f', vmin=0, vmax=1)
    
    plt.title('Model Accuracy by True Class', fontsize=14)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    
    # Save figure
    plt.tight_layout()
    plt.savefig('visualizations/accuracy_by_class.png', dpi=300)
    plt.close()
    
    print("Created prediction analysis visualization")


# -- Generate sentiment score distribution by company

def visualize_sentiment_distribution(sentiment_data=None):

    # If no sentiment data is provided, load it
    if sentiment_data is None:
        sentiment_file = 'data/processed/earnings_sentiment.csv'
        if not os.path.exists(sentiment_file):
            print(f"Sentiment data file not found: {sentiment_file}")
            return
        sentiment_data = pd.read_csv(sentiment_file)
    
    # check - fixed
    if sentiment_data is None or sentiment_data.empty:
        print("No sentiment data available")
        return
    
    # Make box plot of sentiment scores by company
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='ticker', y='sentiment_score', data=sentiment_data)
    plt.title('Sentiment Score Distribution by Company', fontsize=14)
    plt.ylabel('Sentiment Score', fontsize=12)
    plt.xlabel('Company', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('visualizations/sentiment_distribution.png', dpi=300)
    plt.close()
    print("Created sentiment distribution visualization")
    



def main():
    print("Starting visualization generation...")
    
    results = load_results()

    visualize_model_comparison(results['comparison'])
    
    visualize_confusion_matrices(results['confusion'])
    
    visualize_lr_coefficients(results['coefficients'])
    
    create_prediction_analysis(results['predictions'])

    visualize_sentiment_distribution(results)

    
    print("Visualization generation complete! Check 'visualizations/' directory.")

if __name__ == "__main__":
    main()