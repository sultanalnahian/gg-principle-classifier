import networkx as nx
from node2vec import Node2Vec
import re

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

edgeMap = {
				'wanted':'1',
				'needed':'2',
				'seen as':'3',
				'feels':'4',
				'wants':'5',
				'then':'6',
				'feel':'7',
				'want':'8'
				}
##### .edgelist utils and test #####

testArray = ['1 2 3','2 4 0.5']

def edgesToFile( edgeArray, fileName='test.edgelist' ):
	fh = open(fileName,'w')
	for ea in edgeArray:
		fh.write(ea + '\n')
	fh.close()
	return fileName

fn = edgesToFile( testArray )

def fileToEdges( fileName ):
	G = nx.read_edgelist(fileName, nodetype=int, data=(('weight',float),))
	g = G.edges(data=True)
	return g

eg = fileToEdges( fn )

l = ['apple','bat','apple','car','pet','bat','PersonX','food','hungry']

def mapWordsToUniqueIntegers( wordList ):
	d = dict([(y,x+1) for x,y in enumerate(sorted(set(wordList)))])
	return d

mo = mapWordsToUniqueIntegers(l)
#[d[x] for x in wordList]

testCometEdges = {'data':[{'gg':0, 's1':'This is the sentence', 'e1':[('PersonX', 'wants', 'food'),('PersonX', 'feels', 'hungry')], 's2':'This is the second sentence', 'e2':[('PersonX', 'wants', 'money'),('PersonX', 'feels', 'greedy')]}, {'gg':1, 's1':'This is the first  sentence of the second image', 'e1':[('PersonX', 'wants', 'a car'),('PersonX', 'feels', 'adventurous')]} ]}

def extractWordListFromStruct( struct ):	
	st = '{}'.format(struct)
	#an_only = re.sub(r'\W+', '', st)
	an_only = re.sub(r'[^A-Za-z0-9 ]+', '', st)
	return an_only.split()

so = extractWordListFromStruct(testCometEdges)

print(so)

do = mapWordsToUniqueIntegers(so)

def convertCometEdgesToWeightAndFormat( edgeArray, edgeMapping, wordMap ):
	newArray = []
	for triple in edgeArray:
		newStringTriple = '{} {} {}'.format(wordMap[triple[0]],wordMap[triple[2]],edgeMapping[triple[1]]) #edgelist file construction requires strings
		newArray.append(newStringTriple)
	return newArray

wf = convertCometEdgesToWeightAndFormat(testCometEdges['data'][0]['e1'], edgeMap, do ) 

edgesToFile(wf,'test2.edgelist')

##### node2vec utils and test #####

EMBEDDING_FILENAME = './embeddings.emb'
EMBEDDING_MODEL_FILENAME = './embeddings.model'

# Create a graph
graph = nx.read_weighted_edgelist('test2.edgelist')

# Precompute probabilities and generate walks
node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4)

## if d_graph is big enough to fit in the memory, pass temp_folder which has enough disk space
# Note: It will trigger "sharedmem" in Parallel, which will be slow on smaller graphs
#node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4, temp_folder="/mnt/tmp_data")

# Embed
model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)

# Look for most similar nodes
# model.wv.most_similar('2')  # Output node names are always strings

# Save embeddings for later use
model.wv.save_word2vec_format(EMBEDDING_FILENAME)

# Save model for later use
model.save(EMBEDDING_MODEL_FILENAME)

#TODO... for now run code below from project root to test node2vec on generated edgelist file:

# python node2vec/src/main.py --input node2vec/graph/test2.edgelist --output node2vec/emb/test2.emd --weighted
