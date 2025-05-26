1. Change data path in R scripts
2. Run followings:
```
source("compile_jobcorp_12.r")
source("my_job01.r")
source("extract_columns.r")
```
This outputs three files `X1.csv`, `X2.csv`, `D.csv`, `Y.csv`. These corresponds to X1, X2, D1,Y1.

3. To get D2, run followings:
```
source("compile_jobcorp_30.r")
source("my_job01.r")
source("extract_columns.r")
```
Now, `X.csv` and `Y.csv` are identical, but `D.csv` corresponds to D2.

4. To get X2, run followings:
```
source("compile_jobcorp_12.r")
source("my_job01_X2.r")
source("extract_columns_X2.r")
```
Now, `X2.csv` corresponds to X2.

