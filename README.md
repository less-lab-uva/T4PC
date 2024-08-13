# T4PC

## Appendix Tables

### Splits
| Split             | Train   | Val    | Test  |
|-------------------|---------|--------|-------|
| Town 01 left out  | 345,757 | 45,674 | 9,056 |
| Town 02 left out  | 338,732 | 48,040 | 6,690 |
| Town 04 left out  | 361,018 | 47,564 | 7,166 |
| Town 05 left out  | 339,656 | 43,480 | 11,250|
| Town 07 left out  | 366,186 | 45,251 | 9,479 |
| Town 10 left out  | 344,416 | 43,641 | 11,089|

### Preconditions
| Split             | $\phi_1$  | $\phi_2$  | $\phi_3$  | $\phi_4$  | $\phi_5$  | $\phi_6$  |
|-------------------|-----------|-----------|-----------|-----------|-----------|-----------|
| Town 01 left out  | 71,590    | 114,279   | 58,813    | 21,538    | 190,678   | 173,238   |
| Town 02 left out  | 64,462    | 114,947   | 62,016    | 20,310    | 185,042   | 167,602   |
| Town 04 left out  | 82,899    | 124,266   | 51,394    | 24,732    | 220,419   | 205,241   |
| Town 05 left out  | 80,600    | 108,778   | 55,791    | 22,618    | 217,678   | 212,275   |
| Town 07 left out  | 75,898    | 136,267   | 60,898    | 26,541    | 223,841   | 206,229   |
| Town 10 left out  | 75,546    | 127,588   | 55,513    | 23,771    | 224,367   | 210,240   |
| **Average**       | 75,166    | 121,021   | 57,404    | 23,252    | 210,338   | 195,804   |



## Reproduce results

### RQ1, RQ2, and RQ3
1. Download the violations csv files by running:
```bash
python downloader.py --option controlled_experiment_results
```
2. Run the following script to plot the results:
```bash
python controlled_experiment/plot.py
```
Figure 3 in the paper will be generated at `./controlled_experiment/results.png`.

### TCP Controlled Experiment
1. Download the results.json files by running:
```bash
python downloader.py --option case_study_results
```
2. To render the full table with the results, execute the cells in the Jupyter notebook `case_study/results_summary.ipynb`.

