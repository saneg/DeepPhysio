import os
import numpy as np
import keras

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


from keras.models import Sequential
from keras.layers import LSTM, Dense


dataset = []
targets = []
num_classes = 10
num_min_frames = None






# ANALISI E PREPARAZIONE DEI DATI



countNotToInsert = 0


#for fileNPY in os.listdir("/home/gsanesi/VideoFrames/ResNet50_KeyPointsMask/NTURGBD_C001Dataset_SingleBar"):
#for fileNPY in os.listdir("/home/gsanesi/VideoFrames/ResNet50_KeyPointsMask/NTURGBD_C001Dataset_FramesBar"):
#for fileNPY in os.listdir("/home/gsanesi/VideoFrames/ResNet50_KeyPointsMask/NTURGBD_C001Dataset_BestArea_SingleBar"):
#for fileNPY in os.listdir("/home/gsanesi/VideoFrames/ResNet50_KeyPointsMask/NTURGBD_C001Dataset_BestArea_FramesBar"):
for fileNPY in os.listdir("/home/gsanesi/VideoFrames/ResNet50_KeyPointsMask/NTURGBD_C001Dataset_BestScoreAndArea_SingleBar"):
#for fileNPY in os.listdir("/home/gsanesi/VideoFrames/ResNet50_KeyPointsMask/NTURGBD_C001Dataset_BestScoreAndArea_FramesBar"):

    if fileNPY.endswith(".npy"):

        print '\n' + str(fileNPY)

        #videoKeypoints = np.load("/home/gsanesi/VideoFrames/ResNet50_KeyPointsMask/NTURGBD_C001Dataset_SingleBar/" + str(fileNPY) + '')
        #videoKeypoints = np.load("/home/gsanesi/VideoFrames/ResNet50_KeyPointsMask/NTURGBD_C001Dataset_FramesBar/" + str(fileNPY) + '')
        #videoKeypoints = np.load("/home/gsanesi/VideoFrames/ResNet50_KeyPointsMask/NTURGBD_C001Dataset_BestArea_SingleBar/" + str(fileNPY) + '')
        #videoKeypoints = np.load("/home/gsanesi/VideoFrames/ResNet50_KeyPointsMask/NTURGBD_C001Dataset_BestArea_FramesBar/" + str(fileNPY) + '')
        videoKeypoints = np.load("/home/gsanesi/VideoFrames/ResNet50_KeyPointsMask/NTURGBD_C001Dataset_BestScoreAndArea_SingleBar/" + str(fileNPY) + '')
        #videoKeypoints = np.load("/home/gsanesi/VideoFrames/ResNet50_KeyPointsMask/NTURGBD_C001Dataset_BestScoreAndArea_FramesBar/" + str(fileNPY) + '')

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

            reshapedNewVideoKeypoints = np.reshape(newVideoKeypoints, (len(newVideoKeypoints), (np.shape(newVideoKeypoints)[1] * 2)))

            print reshapedNewVideoKeypoints.shape
            #print reshapedNewVideoKeypoints

            dataset.append(reshapedNewVideoKeypoints) #newVideoKeypoints)

            fileNPYName = os.path.splitext(fileNPY)[0]

            if 'A006' in fileNPYName:
                targets.append('pickUp')
            elif 'A007' in fileNPYName:
                targets.append('throw')
            elif 'A008' in fileNPYName:
                targets.append('sittingDown')
            elif 'A009' in fileNPYName:
                targets.append('standingUp')
            elif 'A023' in fileNPYName:
                targets.append('handWaving')
            elif 'A024' in fileNPYName:
                targets.append('kickingSomething')
            elif 'A026' in fileNPYName:
                targets.append('hopping')
            elif 'A027' in fileNPYName:
                targets.append('jumpUp')
            elif 'A031' in fileNPYName:
                targets.append('pointingToSomething')
            elif 'A035' in fileNPYName:
                targets.append('inchino')

        else:
            countNotToInsert = countNotToInsert + 1











# PREPARAZIONE DATASET E TARGETS



print "num_min_frames: " + str(num_min_frames)
print "count_NO_insert: " + str(countNotToInsert)


le = LabelEncoder()
le.fit(targets)
encoded_targets = le.transform(targets)
categorical_targets = keras.utils.to_categorical(encoded_targets, num_classes)

arrDataset = np.array(dataset)
arrCategoricalTargets = np.array(categorical_targets)

print str(arrDataset.shape)
print str(arrCategoricalTargets.shape)


tmpTrain, tmpXTest, tmpTest, tmpYTest = train_test_split(arrDataset, arrCategoricalTargets, test_size=0.3, shuffle=True)

print str(tmpTrain.shape)
print str(tmpTest.shape)
print str(tmpXTest.shape)
print str(tmpYTest.shape)

tmpXTrain, tmpXVal, tmpYTrain, tmpYVal = train_test_split(tmpTrain, tmpTest, test_size=0.2, shuffle=True)

print str(tmpXTrain.shape)
print str(tmpYTrain.shape)
print str(tmpXVal.shape)
print str(tmpYVal.shape)






# SLIDING WINDOW - SU TRAIN SET E SU VALIDATION SET

x_train_tmp = []
y_train_tmp = []

for i in range(len(tmpXTrain)):

    if len(tmpXTrain[i]) == num_min_frames:
        x_train_tmp.append(tmpXTrain[i])
        y_train_tmp.append(tmpYTrain[i])
    else:
        ii = 0
        while ((num_min_frames + ii) <= len(tmpXTrain[i])):
            x_train_tmp.append(tmpXTrain[i][(0 + ii):(num_min_frames + ii)])
            y_train_tmp.append(tmpYTrain[i])
            ii = ii + 1

x_train = np.array(x_train_tmp)
y_train = np.array(y_train_tmp)

print str(x_train.shape)
print str(y_train.shape)


x_val_tmp = []
y_val_tmp = []

for i in range(len(tmpXVal)):

    if len(tmpXVal[i]) == num_min_frames:
        x_val_tmp.append(tmpXVal[i])
        y_val_tmp.append(tmpYVal[i])
    else:
        ii = 0
        while ((num_min_frames + ii) <= len(tmpXVal[i])):
            x_val_tmp.append(tmpXVal[i][(0 + ii):(num_min_frames + ii)])
            y_val_tmp.append(tmpYVal[i])
            ii = ii + 1

x_val = np.array(x_val_tmp)
y_val = np.array(y_val_tmp)

print str(x_val.shape)
print str(y_val.shape)




# MODEL

model = Sequential()
model.add(LSTM(32, input_shape=(num_min_frames, 24)))#17
model.add(Dense(num_classes, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


model.fit(x_train, y_train,
          batch_size=128, epochs=200,
          validation_data=(x_val, y_val))

print '\n'









# SLIDING WINDOW SUL TEST SET

print "\nSliding Window on Test Set\n"

count_correct_classified = 0
num_test_examples = len(tmpXTest)

count_processing = 0

true_confusionMatrix = []
pred_confusionMatrix = []

for i in range(len(tmpXTest)):

    count_processing = count_processing + 1

    print '\n' + str(count_processing) + '/' + str(len(tmpXTest))

    if len(tmpXTest[i]) == num_min_frames:

        x_test = tmpXTest[i].reshape(1, num_min_frames, 24)

        results = model.predict(x_test, batch_size = 1)

        max_value = np.amax(results)
        indexOfMax = np.argmax(results)

        if indexOfMax == np.argmax(tmpYTest[i]):
            count_correct_classified = count_correct_classified + 1
        else:
            print "predicted: " + str(le.inverse_transform([indexOfMax])) + " with " + str(max_value) + " - original: " + str(le.inverse_transform([np.argmax(tmpYTest[i])]))

        pred_confusionMatrix.append(str(le.inverse_transform([indexOfMax])[0]))
        true_confusionMatrix.append(str(le.inverse_transform([np.argmax(tmpYTest[i])])[0]))


    else:
        ii = 0
        top_max_value = None
        top_indexOfMax = None
        while ((num_min_frames + ii) <= len(tmpXTest[i])):

            x_test = tmpXTest[i][(0 + ii):(num_min_frames + ii)].reshape(1, num_min_frames, 24)

            results = model.predict(x_test, batch_size = 1)


            max_value = np.amax(results)
            if top_max_value is None:
                top_max_value = max_value
                top_indexOfMax = np.argmax(results)
            elif max_value > top_max_value:
                top_max_value = max_value
                top_indexOfMax = np.argmax(results)

            ii = ii + 1


        if top_indexOfMax == np.argmax(tmpYTest[i]):
            count_correct_classified = count_correct_classified + 1
        else:
            print "predicted: " + str(le.inverse_transform([top_indexOfMax])) + " with " + str(top_max_value) + " - original: " + str(le.inverse_transform([np.argmax(tmpYTest[i])]))

        pred_confusionMatrix.append(str(le.inverse_transform([top_indexOfMax])[0]))
        true_confusionMatrix.append(str(le.inverse_transform([np.argmax(tmpYTest[i])])[0]))

print "\ncount_correct_classified: " + str(count_correct_classified)
print 'num_test_examples: ' + str(num_test_examples)
print 'test_accuracy: ' + str(float(count_correct_classified)/num_test_examples)

confusionMatrix_SW = confusion_matrix(true_confusionMatrix, pred_confusionMatrix, labels=["pickUp", "throw", "sittingDown", "standingUp", "handWaving", "kickingSomething", "hopping", "jumpUp", "pointingToSomething", "inchino"])

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






print "\nVoting Sliding Window on Test Set\n"

count_correct_classified = 0
num_test_examples = len(tmpXTest)

count_processing = 0

true_confusionMatrix = []
pred_confusionMatrix = []

for i in range(len(tmpXTest)):

    count_processing = count_processing + 1

    print '\n' + str(count_processing) + '/' + str(len(tmpXTest))

    if len(tmpXTest[i]) == num_min_frames:

        x_test = tmpXTest[i].reshape(1, num_min_frames, 24)

        results = model.predict(x_test, batch_size=1)

        max_value = np.amax(results)
        indexOfMax = np.argmax(results)
        if indexOfMax == np.argmax(tmpYTest[i]):
            count_correct_classified = count_correct_classified + 1
        else:
            print "predicted: " + str(le.inverse_transform([indexOfMax])) + " with " + str(
                max_value) + " - original: " + str(le.inverse_transform([np.argmax(tmpYTest[i])]))

        pred_confusionMatrix.append(str(le.inverse_transform([indexOfMax])[0]))
        true_confusionMatrix.append(str(le.inverse_transform([np.argmax(tmpYTest[i])])[0]))

    else:
        ii = 0
        voting = np.zeros((10,), dtype=int)
        predsSWCollection = []

        sum_prediction_proba = None

        while ((num_min_frames + ii) <= len(tmpXTest[i])):
            x_test = tmpXTest[i][(0 + ii):(num_min_frames + ii)].reshape(1, num_min_frames, 24)

            results = model.predict(x_test, batch_size=1)

            if sum_prediction_proba is None:
                sum_prediction_proba = results
            else:
                sum_prediction_proba = [x + y for x, y in zip(sum_prediction_proba, results)]

            max_value = np.amax(results)
            indexOfMax = np.argmax(results)

            voting[indexOfMax] = voting[indexOfMax] + 1
            predsSWCollection.append(indexOfMax)

            ii = ii + 1

        predicted_classes = [indice for indice in range(len(voting)) if voting[indice] == np.amax(voting)]  #prediction = np.argmax(voting)

        if len(predicted_classes) > 1:

            print sum_prediction_proba
            print sum_prediction_proba[0]

            avarage_prediction_proba = [round(float(sum_prediction_proba[0][ind]) / ii, 2) for ind in range(len(sum_prediction_proba[0]))]

            print avarage_prediction_proba

            limitedAvaragePredProba = [avarage_prediction_proba[index] for index in predicted_classes]

            print limitedAvaragePredProba

            prediction = predicted_classes[np.argmax(limitedAvaragePredProba)]
        else:
            prediction = predicted_classes[0]

        if prediction == np.argmax(tmpYTest[i]):
            count_correct_classified = count_correct_classified + 1
        else:
            print "predicted: " + str(le.inverse_transform([prediction])) + " with voting " + str(np.amax(voting)) + "/" + str(ii) + " - original: " + str(le.inverse_transform([np.argmax(tmpYTest[i])]))
            print "voting: " + str(voting)
            print "classes: " + str(le.classes_)

        pred_confusionMatrix.append(str(le.inverse_transform([prediction])[0]))
        true_confusionMatrix.append(str(le.inverse_transform([np.argmax(tmpYTest[i])])[0]))

print "\ncount_correct_classified: " + str(count_correct_classified)
print 'num_test_examples: ' + str(num_test_examples)
print 'test_accuracy: ' + str(float(count_correct_classified)/num_test_examples)

confusionMatrix_SWVoting = confusion_matrix(true_confusionMatrix, pred_confusionMatrix, labels=["pickUp", "throw", "sittingDown", "standingUp", "handWaving", "kickingSomething", "hopping", "jumpUp", "pointingToSomething", "inchino"])

percentage_CM_Voting = []
for i in range(len(confusionMatrix_SWVoting)):
    sumElements = 0
    for j in range(len(confusionMatrix_SWVoting[i])):
        sumElements = sumElements + confusionMatrix_SWVoting[i][j]

    newRow = []
    for j in range(len(confusionMatrix_SWVoting[i])):
        percentage = round((float(confusionMatrix_SWVoting[i][j]) / sumElements) * 100, 2)
        newRow.append(percentage)
    percentage_CM_Voting.append(newRow)


print "\nconfusion matrix: \n" + str(confusionMatrix_SWVoting)
print "\npercentage confusion matrix: \n" + str(np.array(percentage_CM_Voting))

