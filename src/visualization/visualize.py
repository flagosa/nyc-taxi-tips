import matplotlib.pyplot as plt

def plot_monthly_performance(monthly_performance):
    months = list(monthly_performance.keys())
    f1_scores = list(monthly_performance.values())

    plt.figure(figsize=(10, 6))
    plt.bar(months, f1_scores)
    plt.xlabel('Month')
    plt.ylabel('F1 Score')
    plt.title('Monthly Model Performance')
    plt.show()

def plot_ks_results(ks_results, months, features):
    for feature in features:
        ks_stats = [ks_results[month][feature][0] for month in months]
        p_values = [ks_results[month][feature][1] for month in months]
        
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.bar(months, ks_stats)
        plt.title(f'KS Statistic for {feature}')
        plt.xlabel('Month')
        plt.ylabel('KS Statistic')
        
        plt.subplot(2, 1, 2)
        plt.bar(months, p_values)
        plt.title(f'P-values for {feature}')
        plt.xlabel('Month')
        plt.ylabel('P-value')
        
        plt.tight_layout()
        plt.show()