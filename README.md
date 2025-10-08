Current results for personal reference:

==== Summary (aggregated over seeds: 0, 1, 2, 3, 4) ====
No positional encoding (features collapsed to constant):
  Majority baseline  -> Acc 0.759 ± 0.009, Bal Acc 0.500, Boundary Recall 0.000
  Best validation    -> Acc 0.826 ± 0.047, Bal Acc 0.854 ± 0.015, Boundary Recall 0.897 ± 0.051 (epoch 42.600 ± 26.501)
  Test (@best epoch) -> Acc 0.804 ± 0.038, Bal Acc 0.823 ± 0.046, Boundary Recall 0.861 ± 0.162

Tutte embedding replaces missing coordinates:
  Majority baseline  -> Acc 0.759 ± 0.009, Bal Acc 0.500, Boundary Recall 0.000
  Best validation    -> Acc 0.787 ± 0.022, Bal Acc 0.851 ± 0.011, Boundary Recall 0.962 ± 0.024 (epoch 63.000 ± 11.180)
  Test (@best epoch) -> Acc 0.771 ± 0.058, Bal Acc 0.823 ± 0.032, Boundary Recall 0.916 ± 0.029

Spectral embedding replaces missing coordinates:
  Majority baseline  -> Acc 0.759 ± 0.009, Bal Acc 0.500, Boundary Recall 0.000
  Best validation    -> Acc 0.771 ± 0.064, Bal Acc 0.641 ± 0.111, Boundary Recall 0.410 ± 0.236 (epoch 53.800 ± 18.979)
  Test (@best epoch) -> Acc 0.737 ± 0.068, Bal Acc 0.627 ± 0.132, Boundary Recall 0.428 ± 0.273

Force-directed embedding replaces missing coordinates:
  Majority baseline  -> Acc 0.759 ± 0.009, Bal Acc 0.500, Boundary Recall 0.000
  Best validation    -> Acc 0.775 ± 0.014, Bal Acc 0.748 ± 0.023, Boundary Recall 0.702 ± 0.058 (epoch 71.800 ± 3.962)
  Test (@best epoch) -> Acc 0.741 ± 0.033, Bal Acc 0.723 ± 0.059, Boundary Recall 0.691 ± 0.173

Random embedding replaces missing coordinates:
  Majority baseline  -> Acc 0.759 ± 0.009, Bal Acc 0.500, Boundary Recall 0.000
  Best validation    -> Acc 0.719 ± 0.067, Bal Acc 0.731 ± 0.064, Boundary Recall 0.755 ± 0.123 (epoch 70.600 ± 7.701)
  Test (@best epoch) -> Acc 0.698 ± 0.044, Bal Acc 0.693 ± 0.045, Boundary Recall 0.682 ± 0.135
