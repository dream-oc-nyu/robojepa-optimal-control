# E002 reach-v0 results

Seed medians within instance; median across instances. Partial sets are not completed comparisons.

| Planner | Units / 24 | Median progress | Discounted release return | Final success | Median planning s | Overrun ticks |
|---|---:|---:|---:|---:|---:|---:|
| cem_cap10 | 8 | 0.957035 | 3.344587 | 0.0000 | 211.909 | 8 |
| cem_clampexec | 8 | 0.916695 | 3.407427 | 0.0000 | 211.201 | 8 |
| cem_release | 8 | 0.950941 | 2.745897 | 0.0000 | 211.879 | 9 |

| Instance | cem_cap10 | cem_clampexec | cem_release |
|---|---:|---:|---:|
| 100 | 0.788705 | 0.942096 | -0.690445 |
| 101 | 0.980559 | 0.971529 | 0.983738 |
| 102 | 0.974127 | 0.979808 | 0.966024 |
| 103 | 0.947633 | 0.728856 | 0.945667 |
| 104 | 0.976851 | 0.967590 | 0.977208 |
| 105 | 0.926267 | 0.825672 | 0.956215 |
| 106 | 0.966437 | 0.799042 | 0.928880 |
| 107 | 0.168447 | 0.891294 | -0.370965 |

## cem_cap10
rho 5/50/95%: [-0.9099229300233899, 2.4893047141870515, 19.499106445523267] (n=40).
Paired bootstrap versus CEM: `{'paired_instances': 8, 'delta_median': 0.006094222547352923, 'ci95': [-0.029947845975180076, 0.8495408188624898]}`.

## cem_clampexec
rho 5/50/95%: [-0.7780330031271261, 1.647094804378964, 59.14571129092651] (n=40).
Paired bootstrap versus CEM: `{'paired_instances': 8, 'delta_median': -0.03424600451028348, 'ci95': [-0.13858415358431442, 1.2622587497137587]}`.

## cem_release
rho 5/50/95%: [-1.8764339100851088, 1.5268308113880935, 14.989622786880705] (n=40).
Paired bootstrap versus CEM: `None`.
