#!/usr/bin/env python

import sys
import os.path
from os.path import join as PJ
import re
import json
import numpy as np
from tqdm import tqdm
import igraph as ig
import jgf
# import infomap
import math



class NumpyEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
			np.int16, np.int32, np.int64, np.uint8,
			np.uint16, np.uint32, np.uint64)):
			ret = int(obj)
		elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
			ret = float(obj)
		elif isinstance(obj, (np.ndarray,)): 
			ret = obj.tolist()
		else:
			ret = json.JSONEncoder.default(self, obj)

		if isinstance(ret, (float)):
			if math.isnan(ret):
				ret = None

		if isinstance(ret, (bytes, bytearray)):
			ret = ret.decode("utf-8")

		return ret
results = {"errors": [], "warnings": [], "brainlife": [], "datatype_tags": [], "tags": []}

def warning(msg):
	global results
	results['warnings'].append(msg) 
	#results['brainlife'].append({"type": "warning", "msg": msg}) 
	print(msg)

def error(msg):
	global results
	results['errors'].append(msg) 
	#results['brainlife'].append({"type": "error", "msg": msg}) 
	print(msg)

def exitApp():
	global results
	with open("product.json", "w") as fp:
		json.dump(results, fp, cls=NumpyEncoder)
	if len(results["errors"]) > 0:
		sys.exit(1)
	else:
		sys.exit()

def exitAppWithError(msg):
	global results
	results['errors'].append(msg) 
	#results['brainlife'].append({"type": "error", "msg": msg}) 
	print(msg)
	exitApp()



configFilename = "config.json"
argCount = len(sys.argv)
if(argCount > 1):
		configFilename = sys.argv[1]

outputDirectory = "output"
outputFile = PJ(outputDirectory,"network.json.gz")

if(not os.path.exists(outputDirectory)):
		os.makedirs(outputDirectory)

with open(configFilename, "r") as fd:
		config = json.load(fd)


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

networks = jgf.igraph.load(config["network"], compressed=True)

if(len(networks)>1):
	warning("Multiple networks were found in the network data. Null models are being generated only for the first network entry in the list.")

outputNetworks = []

if(len(networks)==0):
	error("No network found in data. Null models requires one network in the file.")
else:	
	network = networks[0]
	for _ in range(nullCount):
		if(modelMethod=="random"):
			gnull = ig.Graph.Erdos_Renyi(n=network.vcount(),m=network.ecount(),directed=network.is_directed())
		elif(modelMethod=="barabasi"):
			gnull = ig.Graph.Barabasi(n=network.vcount(),m=round(0.5*network.ecount()/network.vcount()),directed=network.is_directed())
		elif(modelMethod=="configuration"):
			if(network.is_directed()):
				indegree = network.indegree()
				outdegree = network.outdegree()
				gnull = ig.Graph.Degree_Sequence(outdegree, indegree, method=configurationMethod)
			else:
				gnull = ig.Graph.Degree_Sequence(network.degree(), method=configurationMethod)
		else:
			exitAppWithError("model %s is not valid."%modelMethod)

		if("weight" in network.edge_attributes()):
			if(weightMethod == "sample"):
				gnull.es["weight"] = np.random.choice(network.es['weight'],network.ecount())
			elif(weightMethod == "average"):
				gnull.es["weight"] = np.ones(network.ecount())*(np.mean(network.es['weight']))
		
		
		gnull["null-models"] = nullCount
		gnull["null-models-method"] = modelMethod
		gnull["null-models-weightMethod"] = weightMethod
		gnull["null-models-configurationMethod"] = configurationMethod
		outputNetworks.append(gnull)

jgf.igraph.save(outputNetworks,outputFile, compressed=True)

exitApp()

