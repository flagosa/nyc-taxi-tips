from scipy.stats import ks_2samp

def perform_stat_tests(train_data, test_data, features):
    ks_stats = {}
    for feature in features:
        ks_stat, ks_pvalue = ks_2samp(train_data[feature], test_data[feature])
        ks_stats[feature] = (ks_stat, ks_pvalue)
    return ks_stats
