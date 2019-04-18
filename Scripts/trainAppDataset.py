import os
import numpy as np
import keras
import datetime

from sklearn.preprocessing import LabelEncoder


from keras.models import Sequential
from keras.layers import LSTM, Dense


num_classes = 10
num_min_frames = None



# ----------------------------------------------- TRAIN SET PROCESSING --------------------------------------------

refined_dataset = []
refined_targets_dataset = []

videoTrainNotInsert = 0
videoTrainInsert = 0

countPickUpTrain = 0
countThrowTrain = 0
countSittingDownTrain = 0
countStandingUpTrain = 0
countHandWavingTrain = 0
countKickingSomethingTrain = 0
countHoppingTrain = 0
countJumpUpTrain = 0
countPointingToSomethingTrain = 0
countBowTrain = 0


for fileNPY in os.listdir("/home/gsanesi/PhysioApp/LSTM_Dataset/TrainingSet"):

    if fileNPY.endswith(".npy"):

        print '\n' + str(fileNPY)

        videoKeypoints = np.load("/home/gsanesi/PhysioApp/LSTM_Dataset/TrainingSet/" + str(fileNPY) + '')

        if num_min_frames == None:
            num_min_frames = len(videoKeypoints)
        elif len(videoKeypoints) < num_min_frames:
            num_min_frames = len(videoKeypoints)

        newVideoKeypoints = []

        for i in range(len(videoKeypoints)):
            nvK = []
            for j in range(len(videoKeypoints[i])):

                if j not in [0, 1, 2, 3, 4]:
                    nvK.append(videoKeypoints[i][j])

            newVideoKeypoints.append(nvK)

        print len(newVideoKeypoints)

        notToInsert = False

        for i in range(len(newVideoKeypoints)):
            for j in range(len(newVideoKeypoints[i])):
                if newVideoKeypoints[i][j][0] is None and newVideoKeypoints[i][j][1] is None:

                    keyDistances = [dist[j] for dist in newVideoKeypoints]

                    countNoneKeypoint = 0
                    for elem in range(len(keyDistances)):
                        if keyDistances[elem][0] is None and keyDistances[elem][1] is None:
                            countNoneKeypoint = countNoneKeypoint + 1

                    if (countNoneKeypoint < len(keyDistances) / 3):

                        if i is 0:
                            for jj in range(1, len(keyDistances)):
                                if keyDistances[jj][0] is not None and keyDistances[jj][1] is not None:
                                    newVideoKeypoints[i][j] = keyDistances[jj]
                                    break
                        else:
                            newVideoKeypoints[i][j] = keyDistances[i - 1]

                    else:

                        notToInsert = True
                        pass

        if notToInsert == False:

            reshapedNewVideoKeypoints = np.reshape(newVideoKeypoints,(len(newVideoKeypoints), (np.shape(newVideoKeypoints)[1] * 2)))

            print reshapedNewVideoKeypoints.shape

            refined_dataset.append(reshapedNewVideoKeypoints)

            fileNPYName = os.path.splitext(fileNPY)[0]

            if 'A006' in fileNPYName:
                refined_targets_dataset.append('Pick Up')
                countPickUpTrain = countPickUpTrain + 1
            elif 'A007' in fileNPYName:
                refined_targets_dataset.append('Throw')
                countThrowTrain = countThrowTrain + 1
            elif 'A008' in fileNPYName:
                refined_targets_dataset.append('Sitting Down')
                countSittingDownTrain = countSittingDownTrain + 1
            elif 'A009' in fileNPYName:
                refined_targets_dataset.append('Standing Up')
                countStandingUpTrain = countStandingUpTrain + 1
            elif 'A023' in fileNPYName:
                refined_targets_dataset.append('Hand Waving')
                countHandWavingTrain = countHandWavingTrain + 1
            elif 'A024' in fileNPYName:
                refined_targets_dataset.append('Kicking Something')
                countKickingSomethingTrain = countKickingSomethingTrain + 1
            elif 'A026' in fileNPYName:
                refined_targets_dataset.append('Hopping')
                countHoppingTrain = countHoppingTrain + 1
            elif 'A027' in fileNPYName:
                refined_targets_dataset.append('Jump Up')
                countJumpUpTrain = countJumpUpTrain + 1
            elif 'A031' in fileNPYName:
                refined_targets_dataset.append('Pointing To Something')
                countPointingToSomethingTrain = countPointingToSomethingTrain + 1
            elif 'A035' in fileNPYName:
                refined_targets_dataset.append('Bow')
                countBowTrain = countBowTrain + 1

            videoTrainInsert = videoTrainInsert + 1

        else:
            videoTrainNotInsert = videoTrainNotInsert + 1


print "\nvideo di Train non adeguati e scartati dal Training: " + str(videoTrainNotInsert)
print "\nvideo di Train accettati per il Training: " + str(videoTrainInsert) + '\n'


print 'countPickUpTrain: ' + str(countPickUpTrain)
print 'countThrowTrain: ' + str(countThrowTrain)
print 'countSittingDownTrain: ' + str(countSittingDownTrain)
print 'countStandingUpTrain: ' + str(countStandingUpTrain)
print 'countHandWavingTrain: ' + str(countHandWavingTrain)
print 'countKickingSomethingTrain: ' + str(countKickingSomethingTrain)
print 'countHoppingTrain: ' + str(countHoppingTrain)
print 'countJumpUpTrain: ' + str(countJumpUpTrain)
print 'countPointingToSomethingTrain: ' + str(countPointingToSomethingTrain)
print 'countBowTrain: ' + str(countBowTrain)




# ----------------------------------------------- VALIDATION SET PROCESSING --------------------------------------------


videoValNotInsert = 0
videoValInsert = 0


countPickUpValidation = 0
countThrowValidation = 0
countSittingDownValidation = 0
countStandingUpValidation = 0
countHandWavingValidation = 0
countKickingSomethingValidation = 0
countHoppingValidation = 0
countJumpUpValidation = 0
countPointingToSomethingValidation = 0
countBowValidation = 0





for fileNPY in os.listdir("/home/gsanesi/PhysioApp/LSTM_Dataset/ValidationSet"):

    if fileNPY.endswith(".npy"):

        print '\n' + str(fileNPY)

        videoKeypoints = np.load("/home/gsanesi/PhysioApp/LSTM_Dataset/ValidationSet/" + str(fileNPY) + '')

        if num_min_frames == None:
            num_min_frames = len(videoKeypoints)
        elif len(videoKeypoints) < num_min_frames:
            num_min_frames = len(videoKeypoints)

        newVideoKeypoints = []

        for i in range(len(videoKeypoints)):
            nvK = []
            for j in range(len(videoKeypoints[i])):

                if j not in [0, 1, 2, 3, 4]:
                    nvK.append(videoKeypoints[i][j])

            newVideoKeypoints.append(nvK)

        notToInsert = False

        for i in range(len(newVideoKeypoints)):
            for j in range(len(newVideoKeypoints[i])):
                if newVideoKeypoints[i][j][0] is None and newVideoKeypoints[i][j][1] is None:

                    keyDistances = [dist[j] for dist in newVideoKeypoints]

                    countNoneKeypoint = 0
                    for elem in range(len(keyDistances)):
                        if keyDistances[elem][0] is None and keyDistances[elem][1] is None:
                            countNoneKeypoint = countNoneKeypoint + 1

                    if (countNoneKeypoint < len(keyDistances) / 3):

                        if i is 0:
                            for jj in range(1, len(keyDistances)):
                                if keyDistances[jj][0] is not None and keyDistances[jj][1] is not None:
                                    newVideoKeypoints[i][j] = keyDistances[jj]
                                    break
                        else:
                            newVideoKeypoints[i][j] = keyDistances[i - 1]

                    else:

                        notToInsert = True
                        pass

        if notToInsert == False:

            reshapedNewVideoKeypoints = np.reshape(newVideoKeypoints, (len(newVideoKeypoints), (np.shape(newVideoKeypoints)[1] * 2)))

            print reshapedNewVideoKeypoints.shape

            refined_dataset.append(reshapedNewVideoKeypoints)

            fileNPYName = os.path.splitext(fileNPY)[0]

            if 'A006' in fileNPYName:
                refined_targets_dataset.append('Pick Up')
                countPickUpValidation = countPickUpValidation + 1
            elif 'A007' in fileNPYName:
                refined_targets_dataset.append('Throw')
                countThrowValidation = countThrowValidation + 1
            elif 'A008' in fileNPYName:
                refined_targets_dataset.append('Sitting Down')
                countSittingDownValidation = countSittingDownValidation + 1
            elif 'A009' in fileNPYName:
                refined_targets_dataset.append('Standing Up')
                countStandingUpValidation = countStandingUpValidation + 1
            elif 'A023' in fileNPYName:
                refined_targets_dataset.append('Hand Waving')
                countHandWavingValidation = countHandWavingValidation + 1
            elif 'A024' in fileNPYName:
                refined_targets_dataset.append('Kicking Something')
                countKickingSomethingValidation = countKickingSomethingValidation + 1
            elif 'A026' in fileNPYName:
                refined_targets_dataset.append('Hopping')
                countHoppingValidation = countHoppingValidation + 1
            elif 'A027' in fileNPYName:
                refined_targets_dataset.append('Jump Up')
                countJumpUpValidation = countJumpUpValidation + 1
            elif 'A031' in fileNPYName:
                refined_targets_dataset.append('Pointing To Something')
                countPointingToSomethingValidation = countPointingToSomethingValidation + 1
            elif 'A035' in fileNPYName:
                refined_targets_dataset.append('Bow')
                countBowValidation = countBowValidation + 1

            videoValInsert = videoValInsert + 1

        else:
            videoValNotInsert = videoValNotInsert + 1

print "\nvideo di Validation non adeguati e scartati dal Validation : " + str(videoValNotInsert)
print "\nvideo di Validation accettati per il Validation: " + str(videoValInsert) + '\n'



print 'countPickUpValidation: ' + str(countPickUpValidation)
print 'countThrowValidation: ' + str(countThrowValidation)
print 'countSittingDownValidation: ' + str(countSittingDownValidation)
print 'countStandingUpValidation: ' + str(countStandingUpValidation)
print 'countHandWavingValidation: ' + str(countHandWavingValidation)
print 'countKickingSomethingValidation: ' + str(countKickingSomethingValidation)
print 'countHoppingValidation: ' + str(countHoppingValidation)
print 'countJumpUpValidation: ' + str(countJumpUpValidation)
print 'countPointingToSomethingValidation: ' + str(countPointingToSomethingValidation)
print 'countBowValidation: ' + str(countBowValidation)



# ----------------------------------------------- TEST SET PROCESSING --------------------------------------------



videoTestNotInsert = 0
videoTestInsert = 0



countPickUpTest = 0
countThrowTest = 0
countSittingDownTest = 0
countStandingUpTest = 0
countHandWavingTest = 0
countKickingSomethingTest = 0
countHoppingTest = 0
countJumpUpTest = 0
countPointingToSomethingTest = 0
countBowTest = 0






for fileNPY in os.listdir("/home/gsanesi/PhysioApp/LSTM_Dataset/TestSet"):

    if fileNPY.endswith(".npy"):

        print '\n' + str(fileNPY)

        videoKeypoints = np.load("/home/gsanesi/PhysioApp/LSTM_Dataset/TestSet/" + str(fileNPY) + '')

        if num_min_frames == None:
            num_min_frames = len(videoKeypoints)
        elif len(videoKeypoints) < num_min_frames:
            num_min_frames = len(videoKeypoints)

        newVideoKeypoints = []

        for i in range(len(videoKeypoints)):
            nvK = []
            for j in range(len(videoKeypoints[i])):

                if j not in [0, 1, 2, 3, 4]:
                    nvK.append(videoKeypoints[i][j])

            newVideoKeypoints.append(nvK)

        notToInsert = False

        for i in range(len(newVideoKeypoints)):
            for j in range(len(newVideoKeypoints[i])):
                if newVideoKeypoints[i][j][0] is None and newVideoKeypoints[i][j][1] is None:

                    keyDistances = [dist[j] for dist in newVideoKeypoints]

                    countNoneKeypoint = 0
                    for elem in range(len(keyDistances)):
                        if keyDistances[elem][0] is None and keyDistances[elem][1] is None:
                            countNoneKeypoint = countNoneKeypoint + 1

                    if (countNoneKeypoint < len(keyDistances) / 3):

                        if i is 0:
                            for jj in range(1, len(keyDistances)):
                                if keyDistances[jj][0] is not None and keyDistances[jj][1] is not None:
                                    newVideoKeypoints[i][j] = keyDistances[jj]
                                    break
                        else:
                            newVideoKeypoints[i][j] = keyDistances[i - 1]

                    else:

                        notToInsert = True
                        pass

        if notToInsert == False:

            reshapedNewVideoKeypoints = np.reshape(newVideoKeypoints, (len(newVideoKeypoints), (np.shape(newVideoKeypoints)[1] * 2)))

            print reshapedNewVideoKeypoints.shape

            refined_dataset.append(reshapedNewVideoKeypoints)

            fileNPYName = os.path.splitext(fileNPY)[0]

            if 'A006' in fileNPYName:
                refined_targets_dataset.append('Pick Up')
                countPickUpTest = countPickUpTest + 1
            elif 'A007' in fileNPYName:
                refined_targets_dataset.append('Throw')
                countThrowTest = countThrowTest + 1
            elif 'A008' in fileNPYName:
                refined_targets_dataset.append('Sitting Down')
                countSittingDownTest = countSittingDownTest + 1
            elif 'A009' in fileNPYName:
                refined_targets_dataset.append('Standing Up')
                countStandingUpTest = countStandingUpTest + 1
            elif 'A023' in fileNPYName:
                refined_targets_dataset.append('Hand Waving')
                countHandWavingTest = countHandWavingTest + 1
            elif 'A024' in fileNPYName:
                refined_targets_dataset.append('Kicking Something')
                countKickingSomethingTest = countKickingSomethingTest + 1
            elif 'A026' in fileNPYName:
                refined_targets_dataset.append('Hopping')
                countHoppingTest = countHoppingTest + 1
            elif 'A027' in fileNPYName:
                refined_targets_dataset.append('Jump Up')
                countJumpUpTest = countJumpUpTest + 1
            elif 'A031' in fileNPYName:
                refined_targets_dataset.append('Pointing To Something')
                countPointingToSomethingTest = countPointingToSomethingTest + 1
            elif 'A035' in fileNPYName:
                refined_targets_dataset.append('Bow')
                countBowTest = countBowTest + 1

            videoTestInsert = videoTestInsert + 1

        else:
            videoTestNotInsert = videoTestNotInsert + 1

print "\nvideo di Test non adeguati e scartati dal Test : " + str(videoTestNotInsert)
print "\nvideo di Test accettati per il Test: " + str(videoTestInsert) + '\n'


print "\nnum_min_frames All Sets: " + str(num_min_frames)




print 'countPickUpTest: ' + str(countPickUpTest)
print 'countThrowTest: ' + str(countThrowTest)
print 'countSittingDownTest: ' + str(countSittingDownTest)
print 'countStandingUpTest: ' + str(countStandingUpTest)
print 'countHandWavingTest: ' + str(countHandWavingTest)
print 'countKickingSomethingTest: ' + str(countKickingSomethingTest)
print 'countHoppingTest: ' + str(countHoppingTest)
print 'countJumpUpTest: ' + str(countJumpUpTest)
print 'countPointingToSomethingTest: ' + str(countPointingToSomethingTest)
print 'countBowTest: ' + str(countBowTest)








print '\n\ntotale Video Scartati: ' + str((videoTrainNotInsert + videoValNotInsert + videoTestNotInsert))
print '\ntotale Video Inseriti nel Dataset: ' + str((videoTrainInsert + videoValInsert + videoTestInsert))


print '\ntotale Video Pick Up: ' + str((countPickUpTrain + countPickUpValidation + countPickUpTest))
print '\ntotale Video Throw: ' + str((countThrowTrain + countThrowValidation + countThrowTest))
print '\ntotale Video Sitting Down: ' + str((countSittingDownTrain + countSittingDownValidation + countSittingDownTest))
print '\ntotale Video Standing Up: ' + str((countStandingUpTrain + countStandingUpValidation + countStandingUpTest))
print '\ntotale Video Hand Waving: ' + str((countHandWavingTrain + countHandWavingValidation + countHandWavingTest))
print '\ntotale Video Kicking Something: ' + str((countKickingSomethingTrain + countKickingSomethingValidation + countKickingSomethingTest))
print '\ntotale Video Hopping: ' + str((countHoppingTrain + countHoppingValidation + countHoppingTest))
print '\ntotale Video Jump Up: ' + str((countJumpUpTrain + countJumpUpValidation + countJumpUpTest))
print '\ntotale Video Pointing To Something: ' + str((countPointingToSomethingTrain + countPointingToSomethingValidation + countPointingToSomethingTest))
print '\ntotale Video Bow: ' + str((countBowTrain + countBowValidation + countBowTest))


# ----------------------------------------- PREPARAZIONE SETS E TARGETS -------------------------------------------


le = LabelEncoder()
le.fit(refined_targets_dataset)
encoded_targets_dataset = le.transform(refined_targets_dataset)
categorical_targets_dataset = keras.utils.to_categorical(encoded_targets_dataset, num_classes)

print "\nclasses Train: " + str(le.classes_)

arrRefinedDataset = np.array(refined_dataset)
arrCategoricalTargetsDataset = np.array(categorical_targets_dataset)

tmpXDataset = arrRefinedDataset
tmpYDataset = arrCategoricalTargetsDataset

print 'tmpXDataset.shape: ' + str(tmpXDataset.shape)
print 'tmpYDataset.shape: ' + str(tmpYDataset.shape)






# ----------------------------- SLIDING WINDOW - SU TRAIN SET E SU VALIDATION SET --------------------




x_train_tmp = []
y_train_tmp = []

for i in range(len(tmpXDataset)):

    if len(tmpXDataset[i]) == num_min_frames:
        x_train_tmp.append(tmpXDataset[i])
        y_train_tmp.append(tmpYDataset[i])
    else:
        ii = 0
        while ((num_min_frames + ii) <= len(tmpXDataset[i])):
            x_train_tmp.append(tmpXDataset[i][(0 + ii):(num_min_frames + ii)])
            y_train_tmp.append(tmpYDataset[i])
            ii = ii + 1

x_train = np.array(x_train_tmp)
y_train = np.array(y_train_tmp)

print 'x_train.shape after slicing with num_min_frames: ' + str(x_train.shape)
print 'y_train.shape after slicing with num_min_frames: ' + str(y_train.shape)






# --------------------------------------------------- MODEL TRAINING ----------------------------------------------



model = Sequential()
model.add(LSTM(32, input_shape=(num_min_frames, 24)))  # 17
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


model.fit(x_train, y_train,
          batch_size=128, epochs=200)


now = datetime.datetime.now()
now2 = now.strftime("%Y-%m-%d_%H:%M")
model.save('/home/gsanesi/PhysioApp/Models/model_' + str(now2) + '.h5')

