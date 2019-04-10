# Some utilities for graph processing

Requires Python 3.6+

Clone the repository:

    git clone https://github.com/marcopodda/graphs

Install dependencies:

    pip install -r requirements.txt

And you should be good to go. A simple usage:

    from datasets import load_data, list_datasets

    print(list_datasets())
    graphs = load_data("MUTAG")

If you wish to add more datasets, do as follows:

- create a new config file (in YAML format), following `configs/datasets/.base.conf.yaml` as template
- go to [the repo site](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets) and get the link of the dataset you wish to add.
- compile the YAML file accordingly

Once the new config file is in the `configs/datasets` folder, you can use `load_data`, supplying as argument the name you gave to the dataset.