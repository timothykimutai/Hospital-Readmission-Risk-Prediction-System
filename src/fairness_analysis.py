from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_fairness_metrics(model, X, y, sensitive_attributes):
    results = {}
    
    # Get predictions
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    for attr, protected_values in sensitive_attributes.items():
        attr_values = X[attr]
        metrics = {}
        
        # Calculate metrics for each group
        for group in protected_values:
            mask = attr_values == group
            group_y = y[mask]
            group_y_pred = y_pred[mask]
            group_y_prob = y_prob[mask]
            
            # Skip if group is empty
            if len(group_y) == 0:
                metrics[group] = {
                    'sample_size': 0,
                    'positive_rate': np.nan,
                    'predicted_positive_rate': np.nan,
                    'auroc': np.nan,
                    'fpr': np.nan,
                    'fnr': np.nan
                }
                continue
            
            # Basic metrics
            group_size = len(group_y)
            positive_rate = group_y.mean()
            predicted_positive_rate = group_y_pred.mean()
            
            # Performance metrics
            if len(np.unique(group_y)) > 1:  # Only calculate if both classes exist
                auroc = roc_auc_score(group_y, group_y_prob)
            else:
                auroc = np.nan
                
            # Fairness metrics
            n_neg = (group_y == 0).sum()
            n_pos = (group_y == 1).sum()
            
            if n_neg > 0:
                fpr = ((group_y_pred == 1) & (group_y == 0)).sum() / n_neg
            else:
                fpr = np.nan
                
            if n_pos > 0:
                fnr = ((group_y_pred == 0) & (group_y == 1)).sum() / n_pos
            else:
                fnr = np.nan
                
            metrics[group] = {
                'sample_size': group_size,
                'positive_rate': positive_rate,
                'predicted_positive_rate': predicted_positive_rate,
                'auroc': auroc,
                'fpr': fpr,
                'fnr': fnr
            }
            
        # Calculate fairness disparities
        if len(protected_values) > 1:
            # Find the largest group as reference
            reference_group = max(metrics.items(), key=lambda x: x[1]['sample_size'])[0]
            ref_metrics = metrics[reference_group]
            
            for group in protected_values:
                if group == reference_group:
                    continue
                    
                group_metrics = metrics[group]
                
                # Skip if either group has no samples
                if group_metrics['sample_size'] == 0 or ref_metrics['sample_size'] == 0:
                    metrics[group].update({
                        'demographic_parity_diff': np.nan,
                        'equal_opportunity_diff': np.nan,
                        'average_odds_diff': np.nan
                    })
                    continue
                
                # Demographic parity difference
                dp_diff = (group_metrics['predicted_positive_rate'] - 
                          ref_metrics['predicted_positive_rate'])
                
                # Equal opportunity difference (difference in TPR)
                tpr_diff = ((1 - group_metrics['fnr']) - 
                           (1 - ref_metrics['fnr']))
                
                # Average odds difference (average of FPR and TPR differences)
                avg_odds_diff = 0.5 * (
                    (group_metrics['fpr'] - ref_metrics['fpr']) + 
                    ((1 - group_metrics['fnr']) - (1 - ref_metrics['fnr']))
                )
                
                metrics[group].update({
                    'demographic_parity_diff': dp_diff,
                    'equal_opportunity_diff': tpr_diff,
                    'average_odds_diff': avg_odds_diff
                })
                
        results[attr] = metrics
    return results

def plot_fairness_metrics(fairness_results):
    plots = {}
    
    for attr, metrics in fairness_results.items():
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame.from_dict(metrics, orient='index')
        
        # Create plots for each metric
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Sample sizes
        df['sample_size'].plot(kind='bar', ax=axes[0, 0], title='Sample Sizes')
        axes[0, 0].set_ylabel('Number of Samples')
        
        # Positive rates
        df[['positive_rate', 'predicted_positive_rate']].plot(
            kind='bar', ax=axes[0, 1], title='Actual vs Predicted Positive Rates')
        axes[0, 1].set_ylabel('Rate')
        
        # Performance metrics
        df['auroc'].plot(kind='bar', ax=axes[1, 0], title='AUROC by Group')
        axes[1, 0].set_ylabel('AUROC Score')
        
        # Fairness metrics (if calculated)
        if 'demographic_parity_diff' in df.columns:
            df[['demographic_parity_diff', 'equal_opportunity_diff', 
                'average_odds_diff']].plot(
                kind='bar', ax=axes[1, 1], title='Fairness Metrics')
            axes[1, 1].set_ylabel('Difference from Reference Group')
        
        plt.tight_layout()
        plots[attr] = fig
    return plots
        