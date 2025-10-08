## Current Results (aggregated over seeds: 0, 1, 2, 3, 4)

### Quick Comparison (Best Validation / Test Accuracies)
| Embedding Type        | Best Validation Acc | Test Acc |
|------------------------|---------------------|----------|
| **No Positional**     | 0.826 ± 0.047       | 0.804 ± 0.038 |
| **Tutte**             | 0.787 ± 0.022       | 0.771 ± 0.058 |
| **Spectral**          | 0.771 ± 0.064       | 0.737 ± 0.068 |
| **Force-Directed**    | 0.775 ± 0.014       | 0.741 ± 0.033 |
| **Random**            | 0.719 ± 0.067       | 0.698 ± 0.044 |

---

### No Positional Encoding (features collapsed to constant)
| Metric                | Majority Baseline       | Best Validation                                | Test (@best epoch)                           |
|------------------------|-------------------------|------------------------------------------------|----------------------------------------------|
| Accuracy               | 0.759 ± 0.009          | 0.826 ± 0.047 (epoch 42.600 ± 26.501)         | 0.804 ± 0.038                                |
| Balanced Accuracy      | 0.500                   | 0.854 ± 0.015                                 | 0.823 ± 0.046                                |
| Boundary Recall        | 0.000                   | 0.897 ± 0.051                                 | 0.861 ± 0.162                                |

---

### Tutte Embedding (replaces missing coordinates)
| Metric                | Majority Baseline       | Best Validation                                | Test (@best epoch)                           |
|------------------------|-------------------------|------------------------------------------------|----------------------------------------------|
| Accuracy               | 0.759 ± 0.009          | 0.787 ± 0.022 (epoch 63.000 ± 11.180)         | 0.771 ± 0.058                                |
| Balanced Accuracy      | 0.500                   | 0.851 ± 0.011                                 | 0.823 ± 0.032                                |
| Boundary Recall        | 0.000                   | 0.962 ± 0.024                                 | 0.916 ± 0.029                                |

---

### Spectral Embedding (replaces missing coordinates)
| Metric                | Majority Baseline       | Best Validation                                | Test (@best epoch)                           |
|------------------------|-------------------------|------------------------------------------------|----------------------------------------------|
| Accuracy               | 0.759 ± 0.009          | 0.771 ± 0.064 (epoch 53.800 ± 18.979)         | 0.737 ± 0.068                                |
| Balanced Accuracy      | 0.500                   | 0.641 ± 0.111                                 | 0.627 ± 0.132                                |
| Boundary Recall        | 0.000                   | 0.410 ± 0.236                                 | 0.428 ± 0.273                                |

---

### Force-Directed Embedding (replaces missing coordinates)
| Metric                | Majority Baseline       | Best Validation                                | Test (@best epoch)                           |
|------------------------|-------------------------|------------------------------------------------|----------------------------------------------|
| Accuracy               | 0.759 ± 0.009          | 0.775 ± 0.014 (epoch 71.800 ± 3.962)          | 0.741 ± 0.033                                |
| Balanced Accuracy      | 0.500                   | 0.748 ± 0.023                                 | 0.723 ± 0.059                                |
| Boundary Recall        | 0.000                   | 0.702 ± 0.058                                 | 0.691 ± 0.173                                |

---

### Random Embedding (replaces missing coordinates)
| Metric                | Majority Baseline       | Best Validation                                | Test (@best epoch)                           |
|------------------------|-------------------------|------------------------------------------------|----------------------------------------------|
| Accuracy               | 0.759 ± 0.009          | 0.719 ± 0.067 (epoch 70.600 ± 7.701)          | 0.698 ± 0.044                                |
| Balanced Accuracy      | 0.500                   | 0.731 ± 0.064                                 | 0.693 ± 0.045                                |
| Boundary Recall        | 0.000                   | 0.755 ± 0.123                                 | 0.682 ± 0.135                                |
