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


for fileNPY in os.listdir("/home/gsanesi/VideoFrames/ResNet50_KeyPointsMask/NTURGBD_C001Dataset_BestScoreAndArea_SingleBar"):

    if fileNPY.endswith(".npy"):

        fileNPYName = os.path.splitext(fileNPY)[0]

        if act in fileNPYName:
            print '\n' + str(fileNPY)
            selectedVideos.append(fileNPY)
            targets.append(act)


print ("\nIl numero di video per " + act + " e' " + str(len(selectedVideos)))

arrSelectedVideos = np.array(selectedVideos)
arrTargets = np.array(targets)


trainValVideos, testVideos, trainValTargets, testTargets = train_test_split(arrSelectedVideos, arrTargets, test_size=0.3, shuffle=True)


print ("\nIl numero di video nel train/val set e' " + str(len(trainValVideos)))
print ("\nIl numero di video nel test set e' " + str(len(testVideos)))


for elem in testVideos:

    os.system('cp /home/gsanesi/VideoFrames/ResNet50_KeyPointsMask/NTURGBD_C001Dataset_BestScoreAndArea_SingleBar/' + str(elem) + ' /home/gsanesi/PhysioApp/LSTM_Dataset/TestSet/' + str(elem) + '')



for el in trainValVideos:

    os.system('cp /home/gsanesi/VideoFrames/ResNet50_KeyPointsMask/NTURGBD_C001Dataset_BestScoreAndArea_SingleBar/' + str(el) + ' /home/gsanesi/PhysioApp/TrainValSet/' + str(el) + '')