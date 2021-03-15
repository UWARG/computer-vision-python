import numpy as np

def get_best_location(array):

    tempArrayOne=array[:,[2]].copy()
    tempArrayTwo=array[:,[0]].copy()
    tempArrayThree=array[:,[1]].copy()

#splits the coordinate pairs, error/radius and confidence from the input numpy array into three arrays

    rawArrayOne=[]
    rawArrayTwo=[]
    rawArrayThree=[]

    for i in tempArrayOne:
        for j in list(i):
            rawArrayOne.append(j)
    for i in tempArrayTwo:
        for j in list(i):
            rawArrayTwo.append(j)
    for i in tempArrayThree:
        for j in list(i):
            rawArrayThree.append(j)

    arrayCoords=np.array(rawArrayTwo)
    arrayError=np.array(rawArrayThree)
    arrayConfid=np.array(rawArrayOne)

#fills three numpy arrays with coordinate pairs, error/radius and confidence values

    coordPair=np.average(arrayCoords, axis=0, weights=arrayConfid[:,0])
    weightedError=np.average(arrayError, axis=0, weights=arrayConfid[:,0])

#uses the np.average() function to calculate the weighted average using confidences as weights.
#Formula is sum(weight*value)/sum(weights)

    return(coordPair+(weightedError/2))

#final approximate location is weighted average of coordinate pair values + (weighted average error/2)
#Not sure if there's a better way to handle the error value 
