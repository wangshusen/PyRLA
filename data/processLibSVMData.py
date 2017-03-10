import numpy


def parseRawDataUtil(idxValStr):
    idxValPair = idxValStr[2:-1].split(':') ###
    #idxValPair = idxValStr.split(':') ###
    return [int(idxValPair[0]), float(idxValPair[1])]

def parseRawData(rawData, d):
    labelVec = numpy.zeros(d + 1)
    labelVec[0] = float(rawData[0][2:-1]) ####
    #labelVec[0] = float(rawData[0]) ####
    vecStr = rawData[1:]
    vecIdxVal = list(map(parseRawDataUtil, vecStr))
    for idxValPair in vecIdxVal:
        labelVec[idxValPair[0]] = idxValPair[1]
    return labelVec

def processLibSVMData(inputFileName, d):
    outputFileName = inputFileName + '.npy'
    
    listArrayStr = numpy.loadtxt(inputFileName, dtype='str')
    n = len(listArrayStr)
    mat = numpy.zeros((n, d+1))
    for i in range(n):
        rawData = listArrayStr[i]
        mat[i, :] = parseRawData(rawData, d)
    
    numpy.save(outputFileName, mat)
    return mat

def normalizationTrain(xInputMat, yInputVec):
    yBiasReal = numpy.mean(yInputVec)
    yOutputVec = yInputVec - yBiasReal
    xMaxVec = numpy.max(numpy.abs(xInputMat), axis=0)
    xOutputMat = xInputMat / xMaxVec.reshape(1, len(xMaxVec))
    return xOutputMat, yOutputVec, yBiasReal, xMaxVec

def normalizationTest(xInputMat, yInputVec, yBiasReal, xMaxVec):
    yOutputVec = yInputVec - yBiasReal
    xOutputMat = xInputMat / xMaxVec.reshape(1, len(xMaxVec))
    return xOutputMat, yOutputVec
    

if __name__ == '__main__':
    inputFileName = 'YearPredictionMSD'
    d = 90 # number of features
    processLibSVMData(inputFileName, d)
