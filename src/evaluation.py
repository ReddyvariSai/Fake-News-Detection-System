"""
Model Evaluation Module
Comprehensive evaluation and visualization for fake news detection models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, roc_auc_score,
    matthews_corrcoef, cohen_kappa_score, log_loss,
    brier_score_loss, hamming_loss, jaccard_score
)
from sklearn.calibration import calibration_curve
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """Comprehensive model evaluation and visualization"""
    
    def __init__(self, figsize=(10, 8), style='seaborn-v0_8-darkgrid'):
        """
        Initialize evaluator
        
        Parameters:
        -----------
        figsize : tuple
            Default figure size
        style : str
            Matplotlib style
        """
        plt.style.use(style)
        self.figsize = figsize
        
        # Color palette
        self.colors = {
            'real': '#2ecc71',      # Green
            'fake': '#e74c3c',      # Red
            'correct': '#27ae60',   # Dark green
            'incorrect': '#c0392b', # Dark red
            'neutral': '#3498db',   # Blue
            'highlight': '#f39c12'  # Orange
        }
        
        # Metrics to calculate
        self.metrics_list = [
            'accuracy', 'precision', 'recall', 'f1_score',
            'roc_auc', 'pr_auc', 'mcc', 'kappa',
            'log_loss', 'brier_score'
        ]
    
    def calculate_all_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Calculate all evaluation metrics
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        y_pred_proba : array-like, optional
            Predicted probabilities
            
        Returns:
        --------
        dict : All calculated metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Additional metrics
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        metrics['hamming_loss'] = hamming_loss(y_true, y_pred)
        metrics['jaccard_score'] = jaccard_score(y_true, y_pred)
        
        # Probability-based metrics
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['log_loss'] = log_loss(y_true, y_pred_proba)
            metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba)
            
            # Precision-recall AUC
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            metrics['pr_auc'] = auc(recall, precision)
            metrics['average_precision'] = average_precision_score(y_true, y_pred_proba)
        
        # Class-specific metrics
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics['true_negative'] = tn
        metrics['false_positive'] = fp
        metrics['false_negative'] = fn
        metrics['true_positive'] = tp
        
        # Calculate rates
        metrics['true_negative_rate'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        metrics['true_positive_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Positive and negative predictive values
        metrics['positive_predictive_value'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['negative_predictive_value'] = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        return metrics
    
    def create_classification_report(self, y_true, y_pred, target_names=None):
        """
        Create detailed classification report
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        target_names : list, optional
            Names of target classes
            
        Returns:
        --------
        pandas.DataFrame : Classification report
        """
        if target_names is None:
            target_names = ['Real News', 'Fake News']
        
        report_dict = classification_report(
            y_true, y_pred, 
            target_names=target_names,
            output_dict=True
        )
        
        # Convert to DataFrame
        report_df = pd.DataFrame(report_dict).transpose()
        
        # Add support percentage
        total = report_df.loc['accuracy', 'support']
        for idx in report_df.index:
            if idx not in ['accuracy', 'macro avg', 'weighted avg']:
                report_df.loc[idx, 'support_pct'] = report_df.loc[idx, 'support'] / total * 100
        
        return report_df
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name="Model",
                            normalize=False, save_path=None):
        """
        Plot confusion matrix with detailed annotations
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        model_name : str
            Name of the model for title
        normalize : bool
            Whether to normalize the matrix
        save_path : str, optional
            Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        classes = ['Real News', 'Fake News']
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title_suffix = ' (Normalized)'
        else:
            fmt = 'd'
            title_suffix = ''
        
        plt.figure(figsize=self.figsize)
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=classes, yticklabels=classes,
                   cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
        
        plt.title(f'Confusion Matrix - {model_name}{title_suffix}', 
                 fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        # Add accuracy annotation
        accuracy = accuracy_score(y_true, y_pred)
        plt.text(0.5, -0.15, f'Accuracy: {accuracy:.3f}',
                ha='center', va='center', transform=plt.gca().transAxes,
                fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
        
        # Print detailed confusion matrix
        tn, fp, fn, tp = cm.ravel() if not normalize else cm.ravel() * 100
        
        print(f"\nDetailed Confusion Matrix Analysis:")
        print(f"{'-'*40}")
        print(f"True Negatives (Real → Real):  {tn:{fmt if normalize else ''}}")
        print(f"False Positives (Real → Fake): {fp:{fmt if normalize else ''}}")
        print(f"False Negatives (Fake → Real): {fn:{fmt if normalize else ''}}")
        print(f"True Positives (Fake → Fake):  {tp:{fmt if normalize else ''}}")
        
        if not normalize:
            print(f"\nError Analysis:")
            print(f"  Type I Error (False Positive Rate): {fp/(fp+tn):.3%}")
            print(f"  Type II Error (False Negative Rate): {fn/(fn+tp):.3%}")
    
    def plot_roc_curve(self, y_true, y_pred_proba, model_name="Model", 
                      save_path=None):
        """
        Plot ROC curve with AUC score
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred_proba : array-like
            Predicted probabilities for positive class
        model_name : str
            Name of the model for title
        save_path : str, optional
            Path to save the plot
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Find optimal threshold (Youden's J statistic)
        youden_j = tpr - fpr
        optimal_idx = np.argmax(youden_j)
        optimal_threshold = thresholds[optimal_idx]
        
        plt.figure(figsize=self.figsize)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, color=self.colors['neutral'], lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        
        # Plot optimal point
        plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'o', 
                color=self.colors['highlight'], markersize=10,
                label=f'Optimal threshold = {optimal_threshold:.3f}')
        
        # Plot diagonal
        plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        
        # Customize plot
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'Receiver Operating Characteristic - {model_name}', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Add performance metrics
        plt.text(0.6, 0.2, f'AUC = {roc_auc:.3f}\nOptimal threshold = {optimal_threshold:.3f}',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")
        
        plt.show()
        
        return roc_auc, optimal_threshold
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba, model_name="Model",
                                  save_path=None):
        """
        Plot precision-recall curve
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred_proba : array-like
            Predicted probabilities for positive class
        model_name : str
            Name of the model for title
        save_path : str, optional
            Path to save the plot
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        # Find optimal threshold (F1-score maximization)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else thresholds[-1]
        
        plt.figure(figsize=self.figsize)
        
        # Plot PR curve
        plt.plot(recall, precision, color=self.colors['neutral'], lw=2,
                label=f'PR curve (AP = {avg_precision:.3f})')
        
        # Plot optimal point
        plt.plot(recall[optimal_idx], precision[optimal_idx], 'o',
                color=self.colors['highlight'], markersize=10,
                label=f'Optimal threshold = {optimal_threshold:.3f}')
        
        # Customize plot
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve - {model_name}', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        # Add performance metrics
        plt.text(0.6, 0.2, f'Average Precision = {avg_precision:.3f}\nPR AUC = {pr_auc:.3f}',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Precision-Recall curve saved to {save_path}")
        
        plt.show()
        
        return avg_precision, pr_auc, optimal_threshold
    
    def plot_calibration_curve(self, y_true, y_pred_proba, model_name="Model",
                             n_bins=10, save_path=None):
        """
        Plot calibration curve (reliability diagram)
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred_proba : array-like
            Predicted probabilities
        model_name : str
            Name of the model for title
        n_bins : int
            Number of bins for calibration
        save_path : str, optional
            Path to save the plot
        """
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)
        
        plt.figure(figsize=self.figsize)
        
        # Plot calibration curve
        plt.plot(prob_pred, prob_true, 's-', color=self.colors['neutral'], lw=2,
                label=f'Calibration curve')
        
        # Plot perfect calibration
        plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect calibration')
        
        # Customize plot
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Mean predicted probability', fontsize=12)
        plt.ylabel('Fraction of positives', fontsize=12)
        plt.title(f'Calibration Curve - {model_name}', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc="upper left")
        plt.grid(True, alpha=0.3)
        
        # Calculate Brier score
        brier = brier_score_loss(y_true, y_pred_proba)
        plt.text(0.05, 0.9, f'Brier Score = {brier:.4f}',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Calibration curve saved to {save_path}")
        
        plt.show()
        
        return brier
    
    def plot_feature_importance(self, feature_importance_df, top_n=20, 
                              model_name="Model", save_path=None):
        """
        Plot feature importance
        
        Parameters:
        -----------
        feature_importance_df : pandas.DataFrame
            DataFrame with 'feature' and 'importance' columns
        top_n : int
            Number of top features to plot
        model_name : str
            Name of the model for title
        save_path : str, optional
            Path to save the plot
        """
        # Sort and get top N features
        top_features = feature_importance_df.sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, max(6, top_n * 0.3)))
        
        # Create horizontal bar plot
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
        bars = plt.barh(range(len(top_features)), top_features['importance'], color=colors)
        
        # Customize plot
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance Score', fontsize=12)
        plt.title(f'Top {top_n} Feature Importance - {model_name}', 
                 fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()  # Highest importance on top
        
        # Add value labels
        for bar, importance in zip(bars, top_features['importance']):
            plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                    f'{importance:.4f}', va='center', ha='left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def compare_multiple_models(self, models_dict, X_test, y_test, 
                              save_path=None):
        """
        Compare multiple models
        
        Parameters:
        -----------
        models_dict : dict
            Dictionary of model names and model objects
        X_test : array-like
            Test features
        y_test : array-like
            Test labels
        save_path : str, optional
            Path to save the comparison plot
            
        Returns:
        --------
        pandas.DataFrame : Comparison results
        """
        results = []
        
        for model_name, model in models_dict.items():
            print(f"Evaluating {model_name}...")
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            
            # Calculate metrics
            metrics = self.calculate_all_metrics(y_test, y_pred, y_pred_proba)
            metrics['model'] = model_name
            
            results.append(metrics)
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Sort by F1-score
        results_df = results_df.sort_values('f1_score', ascending=False).reset_index(drop=True)
        
        print(f"\n{'='*60}")
        print("MODEL COMPARISON RESULTS")
        print(f"{'='*60}")
        print(results_df[['model', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']].to_string())
        
        # Plot comparison
        self._plot_model_comparison(results_df, save_path)
        
        return results_df
    
    def _plot_model_comparison(self, results_df, save_path=None):
        """Internal method to plot model comparison"""
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        available_metrics = [m for m in metrics_to_plot if m in results_df.columns]
        
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
        
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(available_metrics):
            ax = axes[idx]
            
            # Sort by metric
            sorted_df = results_df.sort_values(metric, ascending=True)
            
            # Create bar plot
            bars = ax.barh(range(len(sorted_df)), sorted_df[metric], 
                          color=plt.cm.Set3(np.linspace(0, 1, len(sorted_df))))
            
            ax.set_yticks(range(len(sorted_df)))
            ax.set_yticklabels([name.replace('_', ' ').title() for name in sorted_df['model']])
            ax.set_xlabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            
            # Add value labels
            for bar, value in zip(bars, sorted_df[metric]):
                ax.text(value, bar.get_y() + bar.get_height()/2,
                       f'{value:.3f}', va='center', ha='left')
        
        plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model comparison plot saved to {save_path}")
        
        plt.show()
    
    def generate_evaluation_report(self, model, X_test, y_test, model_name="Model",
                                 feature_importance_df=None, save_path='results/evaluation_report.txt'):
        """
        Generate comprehensive evaluation report
        
        Parameters:
        -----------
        model : sklearn model
            Trained model
        X_test : array-like
            Test features
        y_test : array-like
            Test labels
        model_name : str
            Name of the model
        feature_importance_df : pandas.DataFrame, optional
            Feature importance DataFrame
        save_path : str
            Path to save the report
        """
        import os
        from datetime import datetime
        
        # Create results directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Calculate all metrics
        metrics = self.calculate_all_metrics(y_test, y_pred, y_pred_proba)
        
        # Generate classification report
        class_report = self.create_classification_report(y_test, y_pred)
        
        # Write report to file
        with open(save_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("FAKE NEWS DETECTION - MODEL EVALUATION REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Model: {model_name}\n")
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Test Samples: {len(y_test)}\n\n")
            
            f.write("PERFORMANCE METRICS\n")
            f.write("-"*40 + "\n")
            
            # Group metrics
            basic_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            advanced_metrics = ['roc_auc', 'pr_auc', 'mcc', 'kappa']
            probability_metrics = ['log_loss', 'brier_score']
            
            f.write("\nBasic Classification Metrics:\n")
            for metric in basic_metrics:
                if metric in metrics:
                    f.write(f"  {metric.replace('_', ' ').title():25}: {metrics[metric]:.4f}\n")
            
            f.write("\nAdvanced Metrics:\n")
            for metric in advanced_metrics:
                if metric in metrics:
                    f.write(f"  {metric.replace('_', ' ').title():25}: {metrics[metric]:.4f}\n")
            
            if y_pred_proba is not None:
                f.write("\nProbability-based Metrics:\n")
                for metric in probability_metrics:
                    if metric in metrics:
                        f.write(f"  {metric.replace('_', ' ').title():25}: {metrics[metric]:.4f}\n")
            
            f.write("\nConfusion Matrix Details:\n")
            f.write(f"  True Negatives (Real → Real) : {metrics.get('true_negative', 0)}\n")
            f.write(f"  False Positives (Real → Fake): {metrics.get('false_positive', 0)}\n")
            f.write(f"  False Negatives (Fake → Real): {metrics.get('false_negative', 0)}\n")
            f.write(f"  True Positives (Fake → Fake) : {metrics.get('true_positive', 0)}\n\n")
            
            f.write("Error Rates:\n")
            f.write(f"  False Positive Rate: {metrics.get('false_positive_rate', 0):.3%}\n")
            f.write(f"  False Negative Rate: {metrics.get('false_negative_rate', 0):.3%}\n\n")
            
            f.write("CLASSIFICATION REPORT\n")
            f.write("-"*40 + "\n")
            f.write(class_report.to_string())
            
            if feature_importance_df is not None:
                f.write("\n\nTOP 20 FEATURE IMPORTANCE\n")
                f.write("-"*40 + "\n")
                top_features = feature_importance_df.head(20)
                f.write(top_features.to_string())
            
            f.write("\n\n" + "="*70 + "\n")
            f.write("INTERPRETATION GUIDE\n")
            f.write("="*70 + "\n")
            f.write("Accuracy: Overall correctness of predictions\n")
            f.write("Precision: How many predicted 'Fake' are actually fake\n")
            f.write("Recall: How many actual fake news are correctly identified\n")
            f.write("F1-Score: Harmonic mean of precision and recall\n")
            f.write("ROC-AUC: Model's ability to distinguish between classes\n")
            f.write("MCC: Matthews Correlation Coefficient (-1 to 1, 1 is perfect)\n")
            f.write("Kappa: Agreement between predictions and true labels\n\n")
            
            f.write("CONFUSION MATRIX INTERPRETATION:\n")
            f.write("  High FP: Model is too sensitive (classifies real as fake)\n")
            f.write("  High FN: Model is too conservative (misses fake news)\n")
        
        print(f"Evaluation report saved to {save_path}")
        
        # Generate visualizations
        viz_dir = os.path.join(os.path.dirname(save_path), 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Save confusion matrix
        cm_path = os.path.join(viz_dir, f'{model_name}_confusion_matrix.png')
        self.plot_confusion_matrix(y_test, y_pred, model_name, save_path=cm_path)
        
        # Save ROC curve if probabilities are available
        if y_pred_proba is not None:
            roc_path = os.path.join(viz_dir, f'{model_name}_roc_curve.png')
            self.plot_roc_curve(y_test, y_pred_proba, model_name, save_path=roc_path)
            
            pr_path = os.path.join(viz_dir, f'{model_name}_pr_curve.png')
            self.plot_precision_recall_curve(y_test, y_pred_proba, model_name, save_path=pr_path)
        
        # Save feature importance if available
        if feature_importance_df is not None:
            fi_path = os.path.join(viz_dir, f'{model_name}_feature_importance.png')
            self.plot_feature_importance(feature_importance_df, model_name=model_name, save_path=fi_path)
        
        return metrics, class_report

# Usage example
if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, 
                              n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train a model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Evaluate model
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = evaluator.calculate_all_metrics(y_test, y_pred, y_pred_proba)
    print("Metrics:", metrics)
    
    # Generate report
    report = evaluator.generate_evaluation_report(
        model, X_test, y_test, 
        model_name="Random Forest",
        save_path="results/evaluation_report.txt"
    )
