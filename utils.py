import networkx as nx
from node2vec import Node2Vec
import re
import os
import csv

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score

import os
import networkx as nx
import numpy as np
import pandas as pd

import time
#import torch

# from cogdl import experiment
# from cogdl.data import Data
# from cogdl.datasets import BaseDataset, register_dataset
#https://rusty1s.github.io/pycogdl/build/html/notes/create_dataset.html

# edgeMap = {
# 				'wanted':'0.1',
# 				'needed':'0.2',
# 				'seen as':'0.3',
# 				'feels':'0.4',
# 				'wants':'0.5',
# 				'then':'0.6',
# 				'feel':'0.7',
# 				'want':'0.8'
# 				}

#Note: mapped using Nahian's file format

reverseEdgeMap = {
				'wanted':'1',
				'needed':'2',
				'seen as':'3',
				'feels':'4',
				'wants':'5',
				'then':'6',
				'feel':'7',
				'want':'8'
			}

edgeMap = {
				'1','feel'
				'2','then'
				'3','want'
				'4','is'
				'5','feels'
				'6','wants'
				'7','needed'
				'8','then'
				'9','wanted'
			}
principleMap = {
				'attentive':'0',
				'sensibleness':'1',
				'law-abiding':'2',
				'self-care':'3',
				'cooperation':'4',
				'sharing':'4',
				'assistiveness':'5',
				'patience':'6',
				'caution':'7',
				'friendliness':'8',
				'honesty':'9',
				'humility':'10',
				'politeness':'11',
				'cleanliness':'12',
				'respect':'13',
				'enthusiasm':'14',
				'courtesy':'15',
				'responsibility':'16',
				'supportive':'5',
				'attentiveness':'0',
				'empathy ':'17',
				'empathy':'17',
				'confidence': '18',
				'??':'19',
				'???':'19',
				'none':'19'
				}

##### .edgelist utils and test #####

testArray = ['1 2 3','2 4 0.5']

def edgesToFile( edgeArray, fileName='test.edgelist' ):
	fh = open('./data/edges/'+fileName,'w')
	for ea in edgeArray:
		fh.write(ea + '\n')
	fh.close()
	return fileName

def fileToEdges( fileName ):
	G = nx.read_edgelist(fileName, nodetype=int, data=(('weight',float),))
	g = G.edges(data=True)
	return g

def mapWordsToUniqueIntegers( wordList, cometOutput ):
	for c in cometOutput:
		for t in c['tuples']:
			wordList.append(t[2]) #We need to create integers for phrases
	d = dict([(y,x+1) for x,y in enumerate(sorted(set(wordList)))])
	
	return d

testCometEdges = [{'gg':0, 'principle':'patience', 'tuples':[('PersonX', '1', 'food'),('PersonX', '2', 'hungry'),('PersonX', '3', 'money'),('PersonX', '4', 'greedy')]}, {'gg':1, 'principle':'respect', 'tuples':[('PersonX', '5', 'a car'),('PersonX', '6', 'adventurous')]} ]

def extractWordListFromStruct( struct ):	
	st = '{}'.format(struct)
	#an_only = re.sub(r'\W+', '', st)
	an_only = re.sub(r'[^A-Za-z0-9 ]+', '', st)
	return list(set(an_only.split()))


def convertCometEdgesToWeightAndFormat( edgeArray, wordMap ):
	newArray = []
	for triple in edgeArray:
		newStringTriple = '{} {} {}'.format(wordMap[triple[0]],triple[1],wordMap[triple[2]]) #edgelist file construction requires strings
		newArray.append(newStringTriple)
	return newArray

##### node2vec utils and test #####

def embedEdgelist( fileName, dimensions=64, walk_length=30, num_walks=200, workers=16 ):
	name = fileName.replace('.edgelist','')
	print(name)
	EMBEDDING_FILENAME = './data/embeddings/' + name + '.emb'
	#EMBEDDING_MODEL_FILENAME = './data/embeddings/embeddings.model'
	try:
		# Create a graph
		graph = nx.read_weighted_edgelist( './data/edges/'+fileName )
		# Precompute probabilities and generate walks
		node2vec = Node2Vec(graph, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers)
		## if d_graph is big enough to fit in the memory, pass temp_folder which has enough disk space
		# Note: It will trigger "sharedmem" in Parallel, which will be slow on smaller graphs
		#node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4, temp_folder="/mnt/tmp_data")
		# Embed
		model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)

		# Save embeddings for later use
		model.wv.save_word2vec_format(EMBEDDING_FILENAME)
	except Exception as e:
		print(e)
		pass
	# # Save model for later use
	# model.save(EMBEDDING_MODEL_FILENAME)

##### classification #####

actualCometOutput = [] 

template = {'gg':-1, 'principle':'19', 'tuples':[]}
mostRelations = 0

for filename in os.listdir('./data/class-comet-tuples'):
	#print(filename)
	with open('./data/class-comet-tuples/'+filename, 'r') as csvfile:
		# creating a csv reader object
		currentID = filename.replace('.csv','')

		csvreader = csv.reader(csvfile)

		fields = next(csvreader)
		rows = []

		for row in csvreader:
			rows.append(row)

		t = template.copy()
		t['gg'] = currentID

		tups = []
		for r in rows:
			for idf,f in enumerate(r):
				t1 = "They"
				if idf == 0:
					pass
				elif idf == 1:
					if f != '':
						t['principle'] = principleMap[f]
						assert(['principle'] != "")
					pass
				else:
					if idf <= 4:
						t1 = "Others"
						tups.append((t1,idf-1,f))
					else:
						assert(idf != 0)
						tups.append((t1,idf-1,f)) #To account for new principle column

		t['tuples'] = tups
		if len(tups) > mostRelations:
			mostRelations = len(tups)
			#print('New largest set of relations: {}'.format(len(tups))) #Relations range between 30 and 156
		actualCometOutput.append(t)


print('----------------------------------')
so = extractWordListFromStruct(actualCometOutput)
print(so)
print('Number of unique words: {}'.format(len(so)))
print('----------------------------------')
do = mapWordsToUniqueIntegers(so,actualCometOutput)
#print('Word to integer map: {}'.format(do))


start = time.time()

#Formatting for use with COGDL

# n = total number of nodes
# m = total number of edges
# N = number of graphs
# DS_A.txt (m lines): sparse (block diagonal) adjacency matrix for all graphs, each line corresponds to (row, col) resp. (node_id, node_id). All graphs are undirected. Hence, DS_A.txt contains two entries for each edge.
# DS_graph_indicator.txt (n lines): column vector of graph identifiers for all nodes of all graphs, the value in the i-th line is the graph_id of the node with node_id i
# DS_graph_labels.txt (N lines): class labels for all graphs in the data set, the value in the i-th line is the class label of the graph with graph_id i
# DS_node_labels.txt (n lines): column vector of node labels, the value in the i-th line corresponds to the node with node_id i
# There are optional files if the respective information is available:

# DS_edge_labels.txt (m lines; same size as DS_A_sparse.txt): labels for the edges in DS_A_sparse.txt
# DS_edge_attributes.txt (m lines; same size as DS_A.txt): attributes for the edges in DS_A.txt
# DS_node_attributes.txt (n lines): matrix of node attributes, the comma seperated values in the i-th line is the attribute vector of the node with node_id i
# DS_graph_attributes.txt (N lines): regression values for all graphs in the data set, the value in the i-th line is the attribute of the graph with graph_id i

#cogdl = [45, 72, 81, 72, 99, 72, 90, 81, 54, 90, 72, 63, 72, 81, 72, 81, 54, 72, 63, 36, 99, 72, 81, 27, 63, 72, 99, 90, 108, 45, 54, 90, 54, 63, 81, 63, 72, 81, 90, 81, 63, 81, 18, 90, 90, 63, 81, 63, 81, 63, 54, 54, 36, 99, 63, 45, 90, 54, 117, 81, 63, 36, 72, 126, 90, 27, 36, 36, 63, 63, 54, 36, 72, 54, 36, 54, 36, 72, 72, 72, 90, 54, 72, 54, 0, 72, 81, 63, 54, 45, 63, 63, 81, 54, 54, 108, 81, 63, 45, 81, 90, 72, 63, 72, 90, 54, 36, 81, 27, 27, 63, 81, 63, 36, 81, 72, 72, 72, 90, 63, 81, 81, 81, 72, 63, 72, 54, 72, 63, 90, 63, 81, 63, 72, 36, 54, 90, 81, 54, 63, 81, 72, 63, 54, 63, 81, 27, 63, 72, 72, 54, 63, 90, 54, 63, 54, 63, 63, 54, 72, 54, 135, 45, 54, 72, 36, 72, 63, 72, 63, 81, 81, 63, 54, 72, 54, 54, 63, 45, 45, 72, 45, 36, 72, 45, 72, 54, 54, 36, 63, 72, 63, 63, 45, 54, 54, 54, 36, 54, 63, 72, 27, 54, 81, 72, 54, 54, 54, 81, 72, 54, 72, 54, 54, 63, 81, 72, 36, 63, 63, 54, 54, 81, 54, 54, 45, 81, 72, 54, 81, 63, 108, 54, 63, 63, 45, 54, 54, 90, 72, 63, 90, 99, 63, 63, 72, 81, 63, 90, 81, 63, 72, 45, 54, 63, 45, 45, 81, 63, 45, 45, 45, 72, 45, 63, 36, 45, 54, 117, 54, 45, 45, 36, 54, 45, 36, 54, 45, 54, 90, 45, 45, 72, 36, 54, 36, 54, 45, 36, 54, 72, 45, 45, 63, 45, 45, 45, 54, 54, 54, 45, 90, 45, 54, 54, 54, 45, 54, 45, 63, 54, 63, 81, 54, 45, 36, 54, 54, 45, 45, 54, 54, 45, 108, 45, 36, 36, 36, 54, 36, 63, 36, 45, 27, 63, 99, 63, 54, 72, 36, 45, 63, 72, 54, 45, 72, 117, 54, 63, 63, 36, 27, 45, 63, 54, 27, 27, 63, 54, 63, 36, 54, 36, 63, 81, 45, 63, 81, 72, 36, 63, 45, 63, 54, 72, 54, 99, 45, 54, 99, 63, 45, 36, 72, 63, 63, 36, 45, 45, 72, 90, 54, 54, 81, 72, 63, 45, 63, 45, 81, 81, 117, 45, 27, 54, 90, 72, 36, 36, 90, 45, 90, 99, 72, 45, 72, 54, 45, 45, 81, 54, 63, 54, 108, 54, 36, 45, 36, 63, 36, 45, 36, 63, 36, 90, 54, 54, 54, 45, 63, 36, 81, 27, 45, 72, 63, 81, 90, 36, 63, 45, 54, 63, 63, 45, 72, 63, 81, 63, 27, 36, 54, 36, 63, 72, 54, 72, 54, 63, 63, 45, 72, 63, 72, 54, 45, 45, 81, 45, 90, 36, 63, 36, 36, 54, 45, 72, 45, 63, 27, 54, 72, 72, 36, 54, 45, 54, 45, 72, 72, 45, 99, 72, 54, 36, 45, 36, 63, 54, 36, 63, 36, 81, 63, 54, 63, 36, 63, 45, 81, 54, 63, 36, 90, 36, 54, 63, 45, 63, 72, 63, 63, 27, 54, 72, 72, 81, 63, 45, 54, 63, 36, 81, 81, 63, 72, 72, 45, 36, 45, 54, 63, 45, 72, 45, 63, 54, 72, 72, 54, 63, 54, 45, 72, 36, 63, 81, 72, 90, 63, 81, 81, 72, 63, 45, 45, 54, 72, 63, 72, 45, 81, 63, 63, 54, 45, 72, 45, 63, 54, 81, 81, 72, 72, 81, 54, 63, 63, 72, 63, 108, 90, 90, 54, 63, 144, 63, 81, 81, 72, 63, 54, 72, 45, 45, 72, 63, 9, 81, 90, 72, 72, 45, 72, 72, 81, 72, 81, 99, 72, 72, 72, 36, 81, 90, 54, 72, 63, 45, 63, 72, 9, 9, 9, 99, 90, 63, 54, 45, 63, 72, 72, 54, 90, 63, 63, 126, 54, 72, 72, 54, 54, 72, 54, 45, 45, 81, 54, 126, 54, 54, 36, 72, 63, 99, 9, 9, 9, 63, 171, 54, 81, 9, 63, 54, 72, 63, 63, 45, 63, 117, 72, 207, 234, 9, 9, 36, 63, 45, 45, 81, 99, 9, 63, 90, 54, 72, 45, 63, 72, 72, 45, 126, 153, 99, 9, 81, 9, 9, 9, 81, 81, 90, 90, 9, 54, 36, 9, 36, 36, 63, 81, 63, 63, 162, 81, 54, 54, 81, 45, 54, 63, 36, 36, 36, 81, 36, 81, 36, 9, 54, 54, 36, 45, 72, 54, 126, 9, 9, 9, 9, 36, 63, 90, 36, 45, 45, 90, 36, 81, 45, 63, 45, 54, 36, 81, 45, 36, 72, 90, 36, 36, 9, 54, 45, 63, 54, 36, 9, 9, 90, 9, 9, 9, 36, 9, 45, 81, 81, 45, 63, 81, 45, 45, 45, 54, 45, 54, 54, 9, 9, 9, 99, 72, 54, 36, 72, 9, 9, 9, 9, 9, 9, 126, 9, 63, 45, 45, 36, 45, 36, 45, 45, 54, 108, 36, 36, 63, 72, 36, 45, 54, 54, 9, 9, 108, 9, 9, 9, 54, 36, 27, 90, 36, 45, 45, 81, 63, 36, 54, 63, 9, 9, 9, 72, 36, 36, 126, 9, 9, 63, 9, 9, 63, 54, 45, 36, 9, 126, 9, 54, 45, 63, 36, 36, 54, 36, 36, 72, 126, 63, 90, 99, 90, 81, 135, 126, 54, 81]
#print('COGDL Error #:{}'.format(sum(cogdl)))
total1 = 0
total2 = 0
total3 = 0
total4 = 0
total5 = 0
f1 = open('./data/tu-format/principles_A.txt','a')
f2 = open('./data/tu-format/principles_graph_indicator.txt','a')
f3 = open('./data/tu-format/principles_graph_labels.txt','a')
f4 = open('./data/tu-format/principles_node_labels.txt','a')
f5 = open('./data/tu-format/principles_edge_labels.txt','a')

# for i in range(0,len(so)):
# 	f4.write('{}\n'.format(i))
# 	total4 += 1 #Just label the nodes with their id

nodeIDmap = {}
counter = 0
nodeidcounter = 1

for wx in actualCometOutput:
	#print(wx)

	counter += 1
	#print(str(counter))
	wf = convertCometEdgesToWeightAndFormat(wx['tuples'],  do )
	#print(len(wf))
	for edge in wf:
		edg = edge.split(' ')
		if edg[0] in nodeIDmap and edg[2] in nodeIDmap:
			f1.write('{}, {}\n'.format(nodeIDmap[edg[0]],nodeIDmap[edg[2]])) #Adjacency
			total1 += 1
		elif edg[0] in nodeIDmap and edg[2] not in nodeIDmap:
			nodeIDmap[edg[2]] = nodeidcounter
			nodeidcounter += 1
			f2.write('{}\n'.format(counter)) #this new node belongs to which graph?
			f1.write('{}, {}\n'.format(nodeIDmap[edg[0]],nodeIDmap[edg[2]]))
			f4.write('{}\n'.format(edg[2]))
			total1 += 1
			total2 += 1
			total4 += 1
		elif edg[0] not in nodeIDmap and edg[2] in nodeIDmap:
			nodeIDmap[edg[0]] = nodeidcounter
			nodeidcounter += 1
			f2.write('{}\n'.format(counter)) #this new node belongs to which graph?
			f1.write('{}, {}\n'.format(nodeIDmap[edg[0]],nodeIDmap[edg[2]]))
			f4.write('{}\n'.format(edg[0]))
			total1 += 1
			total2 += 1
			total4 += 1
		else:
			nodeIDmap[edg[0]] = nodeidcounter
			f4.write('{}\n'.format(edg[0]))
			nodeidcounter += 1
			nodeIDmap[edg[2]] = nodeidcounter
			f4.write('{}\n'.format(edg[2]))
			nodeidcounter += 1
			f1.write('{}, {}\n'.format(nodeIDmap[edg[0]],nodeIDmap[edg[2]]))
			f2.write('{}\n'.format(counter)) #this new node belongs to which graph?
			f2.write('{}\n'.format(counter)) #this new node belongs to which graph?
			total1 += 1
			total2 += 2
			total4 += 2

		f5.write('{}\n'.format(str(int(edg[1])-1))) #edge attribute
		total5 += 1
	f3.write('{}\n'.format(wx['principle']))
	total3 += 1 #What's the label/class for the graph

	###Uncomment below if you want to generate embeddings for each separate edgelist (probably dont) #########

	#fn = '{}-{}.edgelist'.format(w['gg'],w['principle'])
	#edgesToFile(wf,fn)
	#embedEdgelist(fn)

end = time.time()
print('Generating graphfiles took {} seconds.'.format(end - start))
print('Total writes: All edges:{} Node Graph Indicator:{} Graph Labels:{} Node Labels (Properties.. should match NGI):{} Edge Labels (Properties.. should match AE):{}'.format(total1,total2,total3,total4,total5))
#print('Generating embeddings took {} seconds.'.format(end - start))
f1.close()
f2.close()
f3.close()
f4.close()
f5.close()
#TODO: Convert to compatible graph files (see /data/ subfolder) ... use the TU dataset format (https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets)
#Use COGDL to run experiments

#https://colab.research.google.com/drive/1I8a0DfQ3fI7Njc62__mVXUlcAleUclnb?usp=sharing#scrollTo=_5FBQ9gXpL-W
#https://docs.dgl.ai/tutorials/blitz/2_dglgraph.html
#https://docs.dgl.ai/tutorials/blitz/6_load_data.html#sphx-glr-tutorials-blitz-6-load-data-py
#https://github.com/THUDM/cogdl/blob/master/examples/custom_dataset.py

# TU dataset
# .. automodule:: cogdl.datasets.tu_data
#     :members:
#     :undoc-members:
#     :show-inheritance:



#Associate principle for that set of discriptions with the newly created vector, save to file
#run train.py (read in embeddings)
# # basic usage
# experiment(task="node_classification", dataset="cora", model="gcn")

# # set other hyper-parameters
# experiment(task="node_classification", dataset="cora", model="gcn", hidden_size=32, max_epoch=200)

# # run over multiple models on different seeds
# experiment(task="node_classification", dataset="cora", model=["gcn", "gat"], seed=[1, 2])

# # automl usage
# def func_search(trial):
#     return {
#         "lr": trial.suggest_categorical("lr", [1e-3, 5e-3, 1e-2]),
#         "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
#         "dropout": trial.suggest_uniform("dropout", 0.5, 0.8),
#     }

#CustomizedNodeClassificationDataset(BaseDataset)
#experiment(task="node_classification", dataset="cora", model="gcn", seed=[1, 2], func_search=func_search)
# import numpy as np

# y_true = np.array([[0,1,0],
#                    [0,1,1],
#                    [1,0,1],
#                    [0,0,1]])

# y_pred = np.array([[0,1,1],
#                    [0,1,1],
#                    [0,1,0],
#                    [0,0,0]])

# def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
#     '''
#     Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
#     https://stackoverflow.com/q/32239577/395857
#     '''
#     acc_list = []
#     for i in range(y_true.shape[0]):
#         set_true = set( np.where(y_true[i])[0] )
#         set_pred = set( np.where(y_pred[i])[0] )
#         #print('\nset_true: {0}'.format(set_true))
#         #print('set_pred: {0}'.format(set_pred))
#         tmp_a = None
#         if len(set_true) == 0 and len(set_pred) == 0:
#             tmp_a = 1
#         else:
#             tmp_a = len(set_true.intersection(set_pred))/\
#                     float( len(set_true.union(set_pred)) )
#         #print('tmp_a: {0}'.format(tmp_a))
#         acc_list.append(tmp_a)
#     return np.mean(acc_list)

# if __name__ == "__main__":
#     print('Hamming score: {0}'.format(hamming_score(y_true, y_pred))) # 0.375 (= (0.5+1+0+0)/4)

#     # For comparison sake:
#     import sklearn.metrics

#     # Subset accuracy
#     # 0.25 (= 0+1+0+0 / 4) --> 1 if the prediction for one sample fully matches the gold. 0 otherwise.
#     print('Subset accuracy: {0}'.format(sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)))

#     # Hamming loss (smaller is better)
#     # $$ \text{HammingLoss}(x_i, y_i) = \frac{1}{|D|} \sum_{i=1}^{|D|} \frac{xor(x_i, y_i)}{|L|}, $$
#     # where
#     #  - \\(|D|\\) is the number of samples  
#     #  - \\(|L|\\) is the number of labels 
#     #  - \\(y_i\\) is the ground truth  
#     #  - \\(x_i\\)  is the prediction.  
#     # 0.416666666667 (= (1+0+3+1) / (3*4) )
#     print('Hamming loss: {0}'.format(sklearn.metrics.hamming_loss(y_true, y_pred))) 

# Let
# @misc{KKMMN2016,
#   title  = {Benchmark Data Sets for Graph Kernels},
#   author = {Kristian Kersting and Nils M. Kriege and Christopher Morris and Petra Mutzel and Marion Neumann},
#   year   = {2016},
#   url    = {http://graphkernels.cs.tu-dortmund.de}
# }
