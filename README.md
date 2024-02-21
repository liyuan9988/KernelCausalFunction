# KernelCausalFunction 

Code for ["Kernel Methods for Causal Functions: Dose, Heterogeneous, and Incremental Response Curves"](https://academic.oup.com/biomet/advance-article-abstract/doi/10.1093/biomet/asad042/7219715?redirectedFrom=fulltext) (Biometrica2023) 


## How to run

1. install dependency
```commandline
pip install -r requirements.txt
```

2. add folders
```commandline
mkdir data
mkdir dumps
mkdir logs
```

3. run with configs and commands
```commandline
python main.py configs/<config_name>.json <command_name>
```

4. The result will be found in the `dumps` folder.