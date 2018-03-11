import numpy as np
import operator

def apply_kNN(pvX, pmDataSet, pvClasses, pK):
    ldVotes = {}
    for kClass in pvClasses:
        ldVotes[kClass] = 0
    lnDataSetSize = pmDataSet.shape[0]
    lvDist = ((np.tile(pvX, (lnDataSetSize, 1)) - pmDataSet)**2).sum(1)**0.5
    lvDist = lvDist.argsort()
    for i in range(pK):
        lkClass = pvClasses[lvDist[i]]
        ldVotes[lkClass] = ldVotes[lkClass] + 1
    result = max(ldVotes.items(), key=operator.itemgetter(1))[0]
    return result

def apply_kNN_M(pvX, pmDataSet, pvClasses, pK):
    ldVotes = {}
    for kClass in pvClasses:
        ldVotes[kClass] = 0
    lnDataSetSize = pmDataSet.shape[0]
    lvDist = ((np.tile(pvX, (lnDataSetSize, 1)) - pmDataSet)**2).sum(1)**0.5
    lvDist = lvDist.argsort()
    for i in range(pK):
        lkClass = pvClasses[lvDist[i]]
        ldVotes[lkClass] = ldVotes[lkClass] + lvDist[i]
    result = min(ldVotes.items(), key=operator.itemgetter(1))[0]
    return result