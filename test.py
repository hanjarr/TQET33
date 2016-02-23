from amrafile import amrafile as af
from amracommon.analysis.registration import normalized_cross_correlation
from os import listdir
from os.path import isfile, join
import numpy as np


targetPath = '/moria/data/DB/0030/0015A/wholebody_normalized_water_1_0015A.amra'
prototypePath = '/home/hannes/DB/0030/0015A/prototypes'
prototypes = [join(prototypePath,f) for f in listdir(prototypePath)  if isfile(join(prototypePath, f))]

target = af.parse(targetPath)

prototypeSignals = []
for prototype in prototypes:
	prototypeSignals.append(af.parse(prototype))

prototypesT9, prototypesLF, prototypesRF = [], [], []

for signal in prototypeSignals:
	prototypesT9.append(signal.get_poi('T9'))
	prototypesLF.append(signal.get_poi('LeftFemur'))
	prototypesRF.append(signal.get_poi('RightFemur'))

numpyTarget = target.data
numpyPrototypes = [prototype.data for prototype in prototypeSignals]

ncc = [normalized_cross_correlation(prototype,numpyTarget) for prototype in numpyPrototypes]

maxIndex = ncc.index(max(ncc))

print(maxIndex)

npPrototypesT9 = np.array(prototypesT9)
npPrototypesLF = np.array(prototypesLF)
npPrototypesRF = np.array(prototypesRF)

print(npPrototypesRF[maxIndex])
print(target.get_poi('RightFemur'))

meanT9 = npPrototypesT9.mean(axis = 0)
meanLF = npPrototypesLF.mean(axis = 0)
meanRF = npPrototypesRF.mean(axis = 0)

# print(meanT9)
# print(meanLF)
# print(meanRF)