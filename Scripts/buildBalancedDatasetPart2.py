import os
import numpy as np

from sklearn.model_selection import train_test_split

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt





actions = ['A006', 'A007', 'A008', 'A009', 'A023', 'A024', 'A026', 'A027', 'A031', 'A035']


act = actions[9]

selectedVideos = []
targets = []


for fileNPY in os.listdir("/home/gsanesi/PhysioApp/TrainValSet"):

#for fileNPY in os.listdir("/home/gsanesi/PhysioApp/TrainValSet2"):

    if fileNPY.endswith(".npy"):

        fileNPYName = os.path.splitext(fileNPY)[0]

        if act in fileNPYName:
            print '\n' + str(fileNPY)
            selectedVideos.append(fileNPY)
            targets.append(act)


print ("\nIl numero di video di train/val per " + act + " e' " + str(len(selectedVideos)))

arrSelectedVideos = np.array(selectedVideos)
arrTargets = np.array(targets)


trainVideos, valVideos, trainTargets, valTargets = train_test_split(arrSelectedVideos, arrTargets, test_size=0.2, shuffle=True)


print ("\nIl numero di video nel train set e' " + str(len(trainVideos)))
print ("\nIl numero di video nel val set e' " + str(len(valVideos)))


for elem in valVideos:

    os.system('cp /home/gsanesi/PhysioApp/TrainValSet/' + str(elem) + ' /home/gsanesi/PhysioApp/LSTM_Dataset/ValidationSet/' + str(elem) + '')
    #os.system('cp /home/gsanesi/PhysioApp/TrainValSet2/' + str(elem) + ' /home/gsanesi/PhysioApp/LSTM_Dataset2/ValidationSet/' + str(elem) + '')



for el in trainVideos:

    os.system('cp /home/gsanesi/PhysioApp/TrainValSet/' + str(el) + ' /home/gsanesi/PhysioApp/LSTM_Dataset/TrainingSet/' + str(el) + '')
    #os.system('cp /home/gsanesi/PhysioApp/TrainValSet2/' + str(el) + ' /home/gsanesi/PhysioApp/LSTM_Dataset2/TrainingSet/' + str(el) + '')