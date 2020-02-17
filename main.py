#!/usr/bin/env python

import sys
import os.path
import re
import numpy as np
from tqdm import tqdm

def loadEdgesList(filename):
	edgesList = []
	with open(filename,"r") as fd:
		for line in fd:
			line = line.strip()
			if line:
				fromIndex, toIndex = [int(entry) for entry in re.split(r'\W+',line)]
				edgesList.append((fromIndex,toIndex))
	return edgesList

def saveEdgesListTo(edgesList,filename):
	with open(filename,"w") as fd:
		fd.write("\n".join(["%s\t%s"%edge for edge in edgesList]))


networkName = "sample.edgeslist"
realizations = 100
alpha = 1.0

argCount = len(sys.argv)
if(argCount>1):
	networkName = sys.argv[1]
	if(argCount>2):
		realizations = int(sys.argv[2])
	if(argCount>3):
		alpha = float(sys.argv[3])


networkBaseName = os.path.splitext(os.path.basename(networkName))[0]
outputDirectory = "output"
if(not os.path.exists(outputDirectory)):
	os.makedirs(outputDirectory)

originalEdges = loadEdgesList(networkName)
for realization in tqdm(range(realizations)):
	edgesCount = len(originalEdges)
	selectedEdges = list(np.where(np.random.random(edgesCount)<alpha)[0])
	np.random.shuffle(selectedEdges)
	newEdges = originalEdges.copy()
	if(len(selectedEdges)>2):
		for edgeIndex in selectedEdges:
			while(True):
				selectedEdge = newEdges[edgeIndex]
				crossIndex = np.random.choice(selectedEdges); #selected from selected randomly
				crossEdge = newEdges[crossIndex]

				selectedSource = min(selectedEdge)
				selectedTarget = max(crossEdge)
				crossSource = min(crossEdge)
				crossTarget = max(selectedEdge)

				if(selectedSource!=selectedTarget and crossSource!=crossTarget):
					newEdges[edgeIndex] = (selectedSource,selectedTarget)
					newEdges[crossIndex] = (crossSource,crossTarget)
					break
	
	saveEdgesListTo(newEdges,os.path.join(outputDirectory, "%s-nullA%.3f-%d.edgeslist"%(networkBaseName,alpha,realization)))

