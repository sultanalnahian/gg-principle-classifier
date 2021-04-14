import networkx as nx
from node2vec import Node2Vec
import re
import os
import csv

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
				}
##### .edgelist utils and test #####

testArray = ['1 2 3','2 4 0.5']

def edgesToFile( edgeArray, fileName='test.edgelist' ):
	fh = open(fileName,'w')
	for ea in edgeArray:
		fh.write(ea + '\n')
	fh.close()
	return fileName

#fn = edgesToFile( testArray )

def fileToEdges( fileName ):
	G = nx.read_edgelist(fileName, nodetype=int, data=(('weight',float),))
	g = G.edges(data=True)
	return g

#eg = fileToEdges( fn )

#l = ['apple','bat','apple','car','pet','bat','PersonX','food','hungry']

def mapWordsToUniqueIntegers( wordList, cometOutput ):
	for c in cometOutput:
		for t in c['tuples']:
			wordList.append(t[2]) #We need to create integers for phrases
	d = dict([(y,x+1) for x,y in enumerate(sorted(set(wordList)))])
	
	return d

#mo = mapWordsToUniqueIntegers(l)
#[d[x] for x in wordList]

testCometEdges = [{'gg':0, 'principle':'patience', 'tuples':[('PersonX', '1', 'food'),('PersonX', '2', 'hungry'),('PersonX', '3', 'money'),('PersonX', '4', 'greedy')]}, {'gg':1, 'principle':'respect', 'tuples':[('PersonX', '5', 'a car'),('PersonX', '6', 'adventurous')]} ]

def extractWordListFromStruct( struct ):	
	st = '{}'.format(struct)
	#an_only = re.sub(r'\W+', '', st)
	an_only = re.sub(r'[^A-Za-z0-9 ]+', '', st)
	return an_only.split()

#so = extractWordListFromStruct(testCometEdges)

#print(so)

#do = mapWordsToUniqueIntegers(so)

def convertCometEdgesToWeightAndFormat( edgeArray, wordMap ):
	newArray = []
	for triple in edgeArray:
		newStringTriple = '{} {} {}'.format(wordMap[triple[0]],triple[1],wordMap[triple[2]]) #edgelist file construction requires strings
		newArray.append(newStringTriple)
	return newArray

#wf = convertCometEdgesToWeightAndFormat(testCometEdges[0]['tuples'],  do ) 

#edgesToFile(wf,'test2.edgelist')

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

##### classification #####

actualCometOutput = [] 
#template = {'gg':-1, 'principle':'patience' 's1':'This is the sentence', 'e1':[('PersonX', 'wants', 'food'),('PersonX', 'feels', 'hungry')]}
template = {'gg':-1, 'principle':'patience', 'tuples':[]}

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

		t = template
		t['gg'] = currentID
		t['principle'] = principleMap["patience"]  #!!!!!!!!!!!##################TODO####################


		tups = []
		for r in rows:
			for idf,f in enumerate(r):
				t1 = "They"
				if idf == 0:
					pass
				else:
					if idf <= 3:
						t1 = "Others"
					if f == 'none':
						pass
					else:
						assert(f != 'none')
						assert(idf != 0)
						tups.append((t1,idf,f))

		t['tuples'] = tups
		actualCometOutput.append(t)

#print(actualCometOutput)
#print('----------------------------------')
so = extractWordListFromStruct(actualCometOutput)
#print(so)
do = mapWordsToUniqueIntegers(so,actualCometOutput)
#print(do)
print(actualCometOutput[0]['tuples'])
wf = convertCometEdgesToWeightAndFormat(actualCometOutput[0]['tuples'],  do ) 
#print(wf)
#edgesToFile(wf,'test3.edgelist')

#TODO: For each GG object received from Nahian (containing gg id, principle classification, sentences and extracted triples):
			#Generate a combined vector embedding of all triples
				#Save in a file with principle,vector_embedding

#		Read the file in
#		Train the classifier (Train/Test/Val split)
