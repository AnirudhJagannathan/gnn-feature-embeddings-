## Current Results (aggregated over seeds: 0, 1, 2, 3, 4) on anchored triangulated planar graphs

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


## Results on Random Planar Graphs (aggregated over seeds: 0, 1, 2, 3, 4) 

### Quick Comparison (Best Validation / Test Accuracies)
| Embedding Type        | Best Validation Acc | Test Acc |
|------------------------|---------------------|----------|
| **No Positional**     | 0.560 ± 0.062       | 0.454 ± 0.031 |
| **Tutte**             | 0.550 ± 0.032       | 0.517 ± 0.034 |
| **Spectral**          | 0.530 ± 0.072       | 0.460 ± 0.040 |
| **Force-Directed**    | 0.567 ± 0.033       | 0.495 ± 0.027 |
| **Random**            | 0.546 ± 0.039       | 0.498 ± 0.045 |

---

### No Positional Encoding (features collapsed to constant)
| Metric                | Majority Baseline       | Best Validation                                | Test (@best epoch)                           |
|------------------------|-------------------------|------------------------------------------------|----------------------------------------------|
| Accuracy               | 0.510 ± 0.004          | 0.560 ± 0.062 (epoch 40.600 ± 11.546)         | 0.454 ± 0.031                                |
| Balanced Accuracy      | 0.500                   | 0.559 ± 0.050                                 | 0.460 ± 0.028                                |
| Boundary Recall        | 0.000                   | 0.388 ± 0.091                                 | 0.312 ± 0.126                                |

---

### Tutte Embedding (replaces missing coordinates)
| Metric                | Majority Baseline       | Best Validation                                | Test (@best epoch)                           |
|------------------------|-------------------------|------------------------------------------------|----------------------------------------------|
| Accuracy               | 0.510 ± 0.004          | 0.550 ± 0.032 (epoch 42.200 ± 31.854)         | 0.517 ± 0.034                                |
| Balanced Accuracy      | 0.500                   | 0.540 ± 0.021                                 | 0.517 ± 0.022                                |
| Boundary Recall        | 0.000                   | 0.611 ± 0.215                                 | 0.415 ± 0.330                                |

---

### Spectral Embedding (replaces missing coordinates)
| Metric                | Majority Baseline       | Best Validation                                | Test (@best epoch)                           |
|------------------------|-------------------------|------------------------------------------------|----------------------------------------------|
| Accuracy               | 0.510 ± 0.004          | 0.530 ± 0.072 (epoch 23.000 ± 22.616)         | 0.460 ± 0.040                                |
| Balanced Accuracy      | 0.500                   | 0.536 ± 0.066                                 | 0.464 ± 0.035                                |
| Boundary Recall        | 0.000                   | 0.527 ± 0.130                                 | 0.435 ± 0.147                                |

---

### Force-Directed Embedding (replaces missing coordinates)
| Metric                | Majority Baseline       | Best Validation                                | Test (@best epoch)                           |
|------------------------|-------------------------|------------------------------------------------|----------------------------------------------|
| Accuracy               | 0.510 ± 0.004          | 0.567 ± 0.033 (epoch 32.800 ± 29.047)         | 0.495 ± 0.027                                |
| Balanced Accuracy      | 0.500                   | 0.570 ± 0.031                                 | 0.494 ± 0.028                                |
| Boundary Recall        | 0.000                   | 0.598 ± 0.053                                 | 0.484 ± 0.049                                |

---

### Random Embedding (replaces missing coordinates)
| Metric                | Majority Baseline       | Best Validation                                | Test (@best epoch)                           |
|------------------------|-------------------------|------------------------------------------------|----------------------------------------------|
| Accuracy               | 0.510 ± 0.004          | 0.546 ± 0.039 (epoch 3.600 ± 4.336)           | 0.498 ± 0.045                                |
| Balanced Accuracy      | 0.500                   | 0.558 ± 0.023                                 | 0.502 ± 0.050                                |
| Boundary Recall        | 0.000                   | 0.587 ± 0.239                                 | 0.482 ± 0.164                                |

