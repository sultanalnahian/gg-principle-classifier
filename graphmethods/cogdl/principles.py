from cogdl import experiment

# basic usage
experiment(task="unsupervised_graph_classification", dataset="principles", model="infograph", batch_size=4, train_num=2000)

#experiment(task="graph_classification", dataset="principles", model="gin", batch_size=5, train_num=898)