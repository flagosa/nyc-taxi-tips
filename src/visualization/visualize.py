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
