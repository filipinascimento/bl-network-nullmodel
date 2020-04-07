#!/usr/bin/env python

import sys
import os.path
import re
import json
import numpy as np
from tqdm import tqdm
import igraph as ig
import louvain
# import infomap


def check_symmetric(a, rtol=1e-05, atol=1e-08):
	return np.allclose(a, a.T, rtol=rtol, atol=atol)

def isFloat(value):
	if(value is None):
		return False
	try:
		numericValue = float(value)
		return np.isfinite(numericValue)
	except ValueError:
		return False

def loadCSVMatrix(filename):
	return np.loadtxt(filename,delimiter=",")


configFilename = "config.json"
argCount = len(sys.argv)
if(argCount > 1):
		configFilename = sys.argv[1]

outputDirectory = "output"
csvOutputDirectory = os.path.join(outputDirectory, "csv")

if(not os.path.exists(outputDirectory)):
		os.makedirs(outputDirectory)

if(not os.path.exists(csvOutputDirectory)):
		os.makedirs(csvOutputDirectory)

with open(configFilename, "r") as fd:
		config = json.load(fd)

# "index": "data/index.json",
# "label": "data/label.json",
# "mode": "data/csv"

indexFilename = config["index"]
labelFilename = config["label"]
CSVDirectory = config["csv"]

modelMethod = "random"
weightMethod = "ignore"
nullCount = 1000
configurationMethod = "simple"

if("method" in config):
	modelMethod = config["method"].lower()

if("weights" in config):
	weightMethod = config["weights"].lower()

if("count" in config and config["count"]):
	nullCount = int(config["count"])

if("configuration-method" in config):
	configurationMethod = config["configuration-method"].lower()


with open(indexFilename, "r") as fd:
	indexData = json.load(fd)

with open(labelFilename, "r") as fd:
	labelData = json.load(fd)


for entry in indexData:
	entryFilename = entry["filename"]

	alreadySigned = ("separated-sign" in entry) and entry["separated-sign"]

	#inputfile,outputfile,signedOrNot
	filenames = [entryFilename]
	baseName,extension = os.path.splitext(entryFilename)

	if(alreadySigned):
		filenames += [baseName+"_negative%s"%(extension)]

	# if("null-models" in entry):
	# 	nullCount = int(entry["null-models"])
	# 	filenames += [baseName+"-null_%d%s"%(i,extension) for i in range(nullCount)]
	# 	if(alreadySigned):
	# 		filenames += [baseName+"_negative-null_%d%s"%(i,extension) for i in range(nullCount)]
	
	entry["null-models"] = nullCount;
	for filename in tqdm(filenames):
		networkBaseName,networkExtension = os.path.splitext(filename)
		nullFilenames = [networkBaseName+"-null_%d%s"%(i,networkExtension) for i in range(nullCount)]
		nullNetworks = []

		adjacencyMatrix = loadCSVMatrix(os.path.join(CSVDirectory, filename))
		directionMode=ig.ADJ_DIRECTED
		weights = adjacencyMatrix
		if(check_symmetric(adjacencyMatrix)):
			directionMode=ig.ADJ_UPPER
			weights = weights[np.triu_indices(weights.shape[0], k = 0)]
		g = ig.Graph.Adjacency((adjacencyMatrix != 0).tolist(), directionMode)
		weighted = False
		if(not ((weights==0) | (weights==1)).all()):
			g.es['weight'] = weights[weights != 0]
			weighted = True
		
		for nullFilename in nullFilenames:
			useDefaultWeights = True;
			if(modelMethod=="random"):
				gnull = ig.Graph.Erdos_Renyi(n=g.vcount(),m=g.ecount(),directed=g.is_directed());
			elif(modelMethod=="barabasi"):
				gnull = ig.Graph.Barabasi(n=g.vcount(),m=round(0.5*g.ecount()/g.vcount()),directed=g.is_directed());
			elif(modelMethod=="configuration"):
				if(g.is_directed()):
					indegree = g.indegree()
					outdegree = g.outdegree()
					gnull = ig.Graph.Degree_Sequence(outdegree, indegree, method=configurationMethod)
				else:
					gnull = ig.Graph.Degree_Sequence(g.degree(), method=configurationMethod)
			else:
				raise ValueError("model %s is not valid."%modelMethod);

			if(weighted and useDefaultWeights):
				if(weightMethod == "sample"):
					gnull.es["weight"] = np.random.choice(g.es['weight'],gnull.ecount());
				elif(weightMethod == "average"):
					gnull.es["weight"] = np.ones(gnull.ecount())*(np.mean(g.es['weight']));

			with open(os.path.join(csvOutputDirectory,os.path.basename(nullFilename)), "w") as fd:
				if("weight" in gnull.edge_attributes()):
					outputData = gnull.get_adjacency(attribute='weight').data
				else:
					outputData = gnull.get_adjacency().data
				np.savetxt(fd,outputData,delimiter=",")
		with open(os.path.join(csvOutputDirectory,os.path.basename(filename)), "w") as fd:
			if(weighted):
				outputData = g.get_adjacency(attribute='weight').data
			else:
				outputData = g.get_adjacency().data
			np.savetxt(fd,outputData,delimiter=",")

with open(os.path.join(outputDirectory,"index.json"), "w") as fd:
	json.dump(indexData,fd)

with open(os.path.join(outputDirectory,"label.json"), "w") as fd:
	json.dump(labelData,fd)

