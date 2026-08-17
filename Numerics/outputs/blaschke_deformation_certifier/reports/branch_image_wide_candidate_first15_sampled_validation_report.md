# Phase 4 first-fifteen branch-image sampled contour validation

Branch-image contour-stable row: N = 600, M = 610, rho = 2.45, r = 2.225974769705636, r_tau = 2.08351238444.
Theorem-level perturbation radius: epsilon_X = 1.164116941400e-17.

The deterministic perturbation radius is theorem-certified. The contour minima and finite counts in this table are sampled diagnostics and are not interval-certified.

| rank | target | centre | expected_multiplicity | finite_eigenvalue_count | radius | s_min_gamma | epsilon_m_gamma | max_numeric_error | sampled_validation_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | alpha^1 | 0.65 | 1 | 1 | 0.11375 | 2.507580e-11 | 4.642392e-07 | 2.220446e-15 | True |
| 2 | alpha^2 | 0.4225 | 1 | 1 | 0.06125 | 7.755190e-12 | 1.501081e-06 | 2.775558e-16 | True |
| 3 | mu^1 | 0.3 | 2 | 2 | 0.0126875 | 7.545155e-13 | 1.542867e-05 | 6.106227e-16 | True |
| 4 | alpha^3 | 0.274625 | 1 | 1 | 0.0126875 | 7.545171e-13 | 1.542864e-05 | 1.443290e-15 | True |
| 5 | alpha^4 | 0.17850625 | 1 | 1 | 0.03123859375 | 3.346104e-12 | 3.479022e-06 | 8.326673e-16 | True |
| 6 | alpha^5 | 0.1160290625 | 1 | 1 | 0.01301453125 | 4.304375e-13 | 2.704497e-05 | 1.970646e-15 | True |
| 7 | mu^2 | 0.09 | 2 | 2 | 0.00729055468749 | 9.415140e-14 | 0.000123643082884 | 1.479372e-14 | True |
| 8 | alpha^6 | 0.075418890625 | 1 | 1 | 0.00729055468751 | 9.415140e-14 | 0.000123643082884 | 2.887968e-14 | True |
| 9 | alpha^7 | 0.0490222789063 | 1 | 1 | 0.00857889880869 | 8.611542e-14 | 0.000135181009483 | 2.178813e-15 | True |
| 10 | alpha^8 | 0.0318644812891 | 1 | 1 | 0.00243224064451 | 3.897393e-15 | 0.00298691224309 | 1.935674e-13 | True |
| 11 | mu^3 | 0.027 | 2 | 2 | 0.00243224064455 | 3.897393e-15 | 0.00298691224309 | 2.347393e-13 | True |
| 12 | alpha^9 | 0.0207119128379 | 1 | 1 | 0.00314404358106 | 5.758134e-15 | 0.0020216914528 | 6.848688e-15 | True |
| 13 | alpha^10 | 0.0134627433446 | 1 | 1 | 0.00235598008539 | 1.176374e-15 | 0.009895808818 | 1.294954e-13 | True |
| 14 | alpha^11 | 0.00875078317401 | 1 | 1 | 0.000325391586858 | 2.348523e-17 | 0.495680416526 | 2.184017e-14 | True |
| 15 | mu^4 | 0.0081 | 2 | 2 | 0.00032539158715 | 2.348523e-17 | 0.495680416526 | 1.642297e-13 | True |

Sampled packet passes: 15/15.
Algebraic eigenvalues covered by the sampled finite counts: 19.
Worst sampled small-gain factor: 0.49568.
Contour interval certification: false.