
import os
import numpy as np

import dtw as DTW

from sklearn import svm
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix



refined_train = []
refined_targets_train = []
num_classes = 10
num_min_frames = None

countNotToInsert = 0

referenceVid = "S001C001P005R001A007"

referenceSequence = None


def norma_2(x, y):
    return np.linalg.norm(x - y)


for fileNPY in os.listdir("/home/gsanesi/PhysioApp/LSTM_Dataset/TrainingSet"):

    if fileNPY.endswith(".npy") and referenceVid in fileNPY:

        print '\n' + str(fileNPY)

        videoKeypoints = np.load("/home/gsanesi/PhysioApp/LSTM_Dataset/TrainingSet/" + str(fileNPY) + '')

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
                                if keyDistances[jj][0] is not None and keyDistances[jj][1]:
                                    newVideoKeypoints[i][j] = keyDistances[jj]
                                    break
                        else:

                            newVideoKeypoints[i][j] = keyDistances[i - 1]

                    else:

                        notToInsert = True

                        pass

        if notToInsert == False:
            reshapedNewVideoKeypoints = np.reshape(newVideoKeypoints, (
            len(newVideoKeypoints), (np.shape(newVideoKeypoints)[1] * 2)))

            referenceSequence = reshapedNewVideoKeypoints

        break









for fileNPY in os.listdir("/home/gsanesi/PhysioApp/LSTM_Dataset/TrainingSet"):

    if fileNPY.endswith(".npy") and referenceVid not in fileNPY:

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
                                if keyDistances[jj][0] is not None and keyDistances[jj][1]:
                                    newVideoKeypoints[i][j] = keyDistances[jj]
                                    break
                        else:

                            newVideoKeypoints[i][j] = keyDistances[i - 1]

                    else:

                        notToInsert = True

                        pass

        if notToInsert == False:

            reshapedNewVideoKeypoints = np.reshape(newVideoKeypoints,
                                                   (len(newVideoKeypoints),
                                                    (np.shape(newVideoKeypoints)[1] * 2)))

            print reshapedNewVideoKeypoints.shape

            DTWVidDifferences = []
            for n_splits in range(1, 6):
                k, m = divmod(len(referenceSequence), n_splits)
                referSequenceSplits = list(referenceSequence[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in xrange(n_splits))

                k, m = divmod(len(reshapedNewVideoKeypoints), n_splits)
                reshNewVidKeySplits = list(reshapedNewVideoKeypoints[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in xrange(n_splits))

                for i in range(n_splits):
                    dist, cost, acc, path = DTW.dtw(referSequenceSplits[i], reshNewVidKeySplits[i], dist=norma_2)  # Manhattan
                    print str(acc[-1, -1])
                    DTWVidDifferences.append(acc[-1, -1])


            print "elemento dataset: " + str(DTWVidDifferences)

            refined_train.append(DTWVidDifferences)

            fileNPYName = os.path.splitext(fileNPY)[0]

            if 'A006' in fileNPYName:
                refined_targets_train.append('Pick Up')
            elif 'A007' in fileNPYName:
                refined_targets_train.append('Throw')
            elif 'A008' in fileNPYName:
                refined_targets_train.append('Sitting Down')
            elif 'A009' in fileNPYName:
                refined_targets_train.append('Standing Up')
            elif 'A023' in fileNPYName:
                refined_targets_train.append('Hand Waving')
            elif 'A024' in fileNPYName:
                refined_targets_train.append('Kicking Something')
            elif 'A026' in fileNPYName:
                refined_targets_train.append('Hopping')
            elif 'A027' in fileNPYName:
                refined_targets_train.append('Jump Up')
            elif 'A031' in fileNPYName:
                refined_targets_train.append('Pointing To Something')
            elif 'A035' in fileNPYName:
                refined_targets_train.append('Bow')

        else:
            countNotToInsert = countNotToInsert + 1

print "num_min_frames TRAIN: " + str(num_min_frames)
print "count_NO_insert TRAIN: " + str(countNotToInsert)





for fileNPY in os.listdir("/home/gsanesi/PhysioApp/LSTM_Dataset/ValidationSet"):

    if fileNPY.endswith(".npy") and referenceVid not in fileNPY:

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
                                if keyDistances[jj][0] is not None and keyDistances[jj][1]:
                                    newVideoKeypoints[i][j] = keyDistances[jj]
                                    break
                        else:

                            newVideoKeypoints[i][j] = keyDistances[i - 1]

                    else:

                        notToInsert = True

                        pass

        if notToInsert == False:

            reshapedNewVideoKeypoints = np.reshape(newVideoKeypoints,
                                                   (len(newVideoKeypoints),
                                                    (np.shape(newVideoKeypoints)[1] * 2)))

            print reshapedNewVideoKeypoints.shape


            DTWVidDifferences = []
            for n_splits in range(1, 6):
                k, m = divmod(len(referenceSequence), n_splits)
                referSequenceSplits = list(referenceSequence[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in xrange(n_splits))

                k, m = divmod(len(reshapedNewVideoKeypoints), n_splits)
                reshNewVidKeySplits = list(reshapedNewVideoKeypoints[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in xrange(n_splits))

                for i in range(n_splits):
                    dist, cost, acc, path = DTW.dtw(referSequenceSplits[i], reshNewVidKeySplits[i], dist=norma_2)  # Manhattan
                    print str(acc[-1, -1])
                    DTWVidDifferences.append(acc[-1, -1])

            print "elemento dataset: " + str(DTWVidDifferences)

            refined_train.append(DTWVidDifferences)

            fileNPYName = os.path.splitext(fileNPY)[0]

            if 'A006' in fileNPYName:
                refined_targets_train.append('Pick Up')
            elif 'A007' in fileNPYName:
                refined_targets_train.append('Throw')
            elif 'A008' in fileNPYName:
                refined_targets_train.append('Sitting Down')
            elif 'A009' in fileNPYName:
                refined_targets_train.append('Standing Up')
            elif 'A023' in fileNPYName:
                refined_targets_train.append('Hand Waving')
            elif 'A024' in fileNPYName:
                refined_targets_train.append('Kicking Something')
            elif 'A026' in fileNPYName:
                refined_targets_train.append('Hopping')
            elif 'A027' in fileNPYName:
                refined_targets_train.append('Jump Up')
            elif 'A031' in fileNPYName:
                refined_targets_train.append('Pointing To Something')
            elif 'A035' in fileNPYName:
                refined_targets_train.append('Bow')

        else:
            countNotToInsert = countNotToInsert + 1

print "num_min_frames TRAIN: " + str(num_min_frames)
print "count_NO_insert TRAIN: " + str(countNotToInsert)






refined_test = []
refined_targets_test = []
num_min_frames = None
countNotToInsert = 0

for fileNPY in os.listdir("/home/gsanesi/PhysioApp/LSTM_Dataset/TestSet"):

    if fileNPY.endswith(".npy") and referenceVid not in fileNPY:

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
                                if keyDistances[jj][0] is not None and keyDistances[jj][1]:
                                    newVideoKeypoints[i][j] = keyDistances[jj]
                                    break
                        else:

                            newVideoKeypoints[i][j] = keyDistances[i - 1]

                    else:

                        notToInsert = True

                        pass

        if notToInsert == False:

            reshapedNewVideoKeypoints = np.reshape(newVideoKeypoints,
                                                   (len(newVideoKeypoints),
                                                    (np.shape(newVideoKeypoints)[1] * 2)))

            print reshapedNewVideoKeypoints.shape

            DTWVidDifferences = []
            for n_splits in range(1, 6):
                k, m = divmod(len(referenceSequence), n_splits)
                referSequenceSplits = list(referenceSequence[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in xrange(n_splits))

                k, m = divmod(len(reshapedNewVideoKeypoints), n_splits)
                reshNewVidKeySplits = list(reshapedNewVideoKeypoints[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in xrange(n_splits))

                for i in range(n_splits):
                    dist, cost, acc, path = DTW.dtw(referSequenceSplits[i], reshNewVidKeySplits[i], dist=norma_2)  # Manhattan
                    print str(acc[-1, -1])
                    DTWVidDifferences.append(acc[-1, -1])

            print "elemento dataset: " + str(DTWVidDifferences)

            refined_test.append(DTWVidDifferences)

            fileNPYName = os.path.splitext(fileNPY)[0]

            if 'A006' in fileNPYName:
                refined_targets_test.append('Pick Up')
            elif 'A007' in fileNPYName:
                refined_targets_test.append('Throw')
            elif 'A008' in fileNPYName:
                refined_targets_test.append('Sitting Down')
            elif 'A009' in fileNPYName:
                refined_targets_test.append('Standing Up')
            elif 'A023' in fileNPYName:
                refined_targets_test.append('Hand Waving')
            elif 'A024' in fileNPYName:
                refined_targets_test.append('Kicking Something')
            elif 'A026' in fileNPYName:
                refined_targets_test.append('Hopping')
            elif 'A027' in fileNPYName:
                refined_targets_test.append('Jump Up')
            elif 'A031' in fileNPYName:
                refined_targets_test.append('Pointing To Something')
            elif 'A035' in fileNPYName:
                refined_targets_test.append('Bow')

        else:
            countNotToInsert = countNotToInsert + 1

print "num_min_frames TEST: " + str(num_min_frames)
print "count_NO_insert TEST: " + str(countNotToInsert)

x_train = np.array(refined_train)
y_train = np.array(refined_targets_train)
x_test = np.array(refined_test)
y_test = np.array(refined_targets_test)

print str(x_train.shape)
print str(y_train.shape)
print str(x_test.shape)
print str(y_test.shape)


#  SVM


clf = svm.SVC(C=1.0, gamma='scale', kernel='linear', decision_function_shape='ovo')
clf.fit(x_train, y_train)


print "\nOne vs One\n"

count_correct_classified = 0
num_test_examples = len(x_test)

count_processing = 0

true_confusionMatrix = []
pred_confusionMatrix = []

for i in range(len(x_test)):

    count_processing = count_processing + 1

    print '\n' + str(count_processing) + '/' + str(len(x_test))

    results = clf.predict([x_test[i]])
    dec = clf.decision_function([x_test[i]])
    print str(dec)

    if results[0] == y_test[i]:
        count_correct_classified = count_correct_classified + 1
    else:
        print "predicted: " + str(results[0]) + " - original: " + str(y_test[i])

    pred_confusionMatrix.append(results[0])
    true_confusionMatrix.append(y_test[i])

print "\ncount_correct_classified: " + str(count_correct_classified)
print 'num_test_examples: ' + str(num_test_examples)
print 'test_accuracy: ' + str(float(count_correct_classified) / num_test_examples)

confusionMatrix_SW = confusion_matrix(true_confusionMatrix, pred_confusionMatrix,
                                      labels=["Pick Up", "Throw", "Sitting Down", "Standing Up", "Hand Waving",
                                              "Kicking Something", "Hopping", "Jump Up", "Pointing To Something", "Bow"])

percentage_CM_SW = []
for i in range(len(confusionMatrix_SW)):
    sumElements = 0
    for j in range(len(confusionMatrix_SW[i])):
        sumElements = sumElements + confusionMatrix_SW[i][j]

    newRow = []
    for j in range(len(confusionMatrix_SW[i])):
        percentage = round((float(confusionMatrix_SW[i][j]) / sumElements) * 100, 2)
        newRow.append(percentage)
    percentage_CM_SW.append(newRow)

print "\nconfusion matrix: \n" + str(confusionMatrix_SW)
print "\npercentage confusion matrix: \n" + str(np.array(percentage_CM_SW))