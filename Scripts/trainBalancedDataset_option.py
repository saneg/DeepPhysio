import os
import numpy as np
import keras
import datetime

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix


from keras.models import Sequential
from keras.layers import LSTM, Dense


num_classes = 10
num_min_frames = None



# ----------------------------------------------- TRAIN SET PROCESSING --------------------------------------------

refined_train = []
refined_targets_train = []

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


for fileNPY in os.listdir("/home/gsanesi/PhysioApp/LSTM_Dataset2/TrainingSet"):

    if fileNPY.endswith(".npy"):

        print '\n' + str(fileNPY)

        videoKeypoints = np.load("/home/gsanesi/PhysioApp/LSTM_Dataset2/TrainingSet/" + str(fileNPY) + '')

        if num_min_frames == None:
            num_min_frames = len(videoKeypoints)
        elif len(videoKeypoints) < num_min_frames:
            num_min_frames = len(videoKeypoints)


        print videoKeypoints.shape

        refined_train.append(videoKeypoints)

        fileNPYName = os.path.splitext(fileNPY)[0]

        if 'A006' in fileNPYName:
            refined_targets_train.append('Pick Up')
            countPickUpTrain = countPickUpTrain + 1
        elif 'A007' in fileNPYName:
            refined_targets_train.append('Throw')
            countThrowTrain = countThrowTrain + 1
        elif 'A008' in fileNPYName:
            refined_targets_train.append('Sitting Down')
            countSittingDownTrain = countSittingDownTrain + 1
        elif 'A009' in fileNPYName:
            refined_targets_train.append('Standing Up')
            countStandingUpTrain = countStandingUpTrain + 1
        elif 'A023' in fileNPYName:
            refined_targets_train.append('Hand Waving')
            countHandWavingTrain = countHandWavingTrain + 1
        elif 'A024' in fileNPYName:
            refined_targets_train.append('Kicking Something')
            countKickingSomethingTrain = countKickingSomethingTrain + 1
        elif 'A026' in fileNPYName:
            refined_targets_train.append('Hopping')
            countHoppingTrain = countHoppingTrain + 1
        elif 'A027' in fileNPYName:
            refined_targets_train.append('Jump Up')
            countJumpUpTrain = countJumpUpTrain + 1
        elif 'A031' in fileNPYName:
            refined_targets_train.append('Pointing To Something')
            countPointingToSomethingTrain = countPointingToSomethingTrain + 1
        elif 'A035' in fileNPYName:
            refined_targets_train.append('Bow')
            countBowTrain = countBowTrain + 1


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



refined_validation = []
refined_targets_validation = []


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





for fileNPY in os.listdir("/home/gsanesi/PhysioApp/LSTM_Dataset2/ValidationSet"):

    if fileNPY.endswith(".npy"):

        print '\n' + str(fileNPY)

        videoKeypoints = np.load("/home/gsanesi/PhysioApp/LSTM_Dataset2/ValidationSet/" + str(fileNPY) + '')

        if num_min_frames == None:
            num_min_frames = len(videoKeypoints)
        elif len(videoKeypoints) < num_min_frames:
            num_min_frames = len(videoKeypoints)



        print videoKeypoints.shape

        refined_validation.append(videoKeypoints)

        fileNPYName = os.path.splitext(fileNPY)[0]

        if 'A006' in fileNPYName:
            refined_targets_validation.append('Pick Up')
            countPickUpValidation = countPickUpValidation + 1
        elif 'A007' in fileNPYName:
            refined_targets_validation.append('Throw')
            countThrowValidation = countThrowValidation + 1
        elif 'A008' in fileNPYName:
            refined_targets_validation.append('Sitting Down')
            countSittingDownValidation = countSittingDownValidation + 1
        elif 'A009' in fileNPYName:
            refined_targets_validation.append('Standing Up')
            countStandingUpValidation = countStandingUpValidation + 1
        elif 'A023' in fileNPYName:
            refined_targets_validation.append('Hand Waving')
            countHandWavingValidation = countHandWavingValidation + 1
        elif 'A024' in fileNPYName:
            refined_targets_validation.append('Kicking Something')
            countKickingSomethingValidation = countKickingSomethingValidation + 1
        elif 'A026' in fileNPYName:
            refined_targets_validation.append('Hopping')
            countHoppingValidation = countHoppingValidation + 1
        elif 'A027' in fileNPYName:
            refined_targets_validation.append('Jump Up')
            countJumpUpValidation = countJumpUpValidation + 1
        elif 'A031' in fileNPYName:
            refined_targets_validation.append('Pointing To Something')
            countPointingToSomethingValidation = countPointingToSomethingValidation + 1
        elif 'A035' in fileNPYName:
            refined_targets_validation.append('Bow')
            countBowValidation = countBowValidation + 1



print 'countPickUpValidation: ' + str(countPickUpValidation)
print 'countThrowValidation: ' + str(countThrowValidation)
print 'countSittingDownValidation: ' + str(countSittingDownValidation)
print 'countStandingUpValidation: ' + str(countStandingUpValidation)
print 'countHandWavingValidation: ' + str(countHandWavingValidation)
print 'countKickingSomethingTest: ' + str(countKickingSomethingValidation)
print 'countHoppingValidation: ' + str(countHoppingValidation)
print 'countJumpUpValidation: ' + str(countJumpUpValidation)
print 'countPointingToSomethingValidation: ' + str(countPointingToSomethingValidation)
print 'countBowValidation: ' + str(countBowValidation)



# ----------------------------------------------- TEST SET PROCESSING --------------------------------------------



refined_test = []
refined_targets_test = []


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






for fileNPY in os.listdir("/home/gsanesi/PhysioApp/LSTM_Dataset2/TestSet"):

    if fileNPY.endswith(".npy"):

        print '\n' + str(fileNPY)

        videoKeypoints = np.load("/home/gsanesi/PhysioApp/LSTM_Dataset2/TestSet/" + str(fileNPY) + '')

        if num_min_frames == None:
            num_min_frames = len(videoKeypoints)
        elif len(videoKeypoints) < num_min_frames:
            num_min_frames = len(videoKeypoints)



        print videoKeypoints.shape

        refined_test.append(videoKeypoints)

        fileNPYName = os.path.splitext(fileNPY)[0]

        if 'A006' in fileNPYName:
            refined_targets_test.append('Pick Up')
            countPickUpTest = countPickUpTest + 1
        elif 'A007' in fileNPYName:
            refined_targets_test.append('Throw')
            countThrowTest = countThrowTest + 1
        elif 'A008' in fileNPYName:
            refined_targets_test.append('Sitting Down')
            countSittingDownTest = countSittingDownTest + 1
        elif 'A009' in fileNPYName:
            refined_targets_test.append('Standing Up')
            countStandingUpTest = countStandingUpTest + 1
        elif 'A023' in fileNPYName:
            refined_targets_test.append('Hand Waving')
            countHandWavingTest = countHandWavingTest + 1
        elif 'A024' in fileNPYName:
            refined_targets_test.append('Kicking Something')
            countKickingSomethingTest = countKickingSomethingTest + 1
        elif 'A026' in fileNPYName:
            refined_targets_test.append('Hopping')
            countHoppingTest = countHoppingTest + 1
        elif 'A027' in fileNPYName:
            refined_targets_test.append('Jump Up')
            countJumpUpTest = countJumpUpTest + 1
        elif 'A031' in fileNPYName:
            refined_targets_test.append('Pointing To Something')
            countPointingToSomethingTest = countPointingToSomethingTest + 1
        elif 'A035' in fileNPYName:
            refined_targets_test.append('Bow')
            countBowTest = countBowTest + 1



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








# ----------------------------------------- PREPARAZIONE SETS E TARGETS -------------------------------------------



le = LabelEncoder()
le.fit(refined_targets_train)
encoded_targets_train = le.transform(refined_targets_train)
categorical_targets_train = keras.utils.to_categorical(encoded_targets_train, num_classes)

print "\nclasses Train: " + str(le.classes_)

le2 = LabelEncoder()
le2.fit(refined_targets_validation)
encoded_targets_val = le2.transform(refined_targets_validation)
categorical_targets_val = keras.utils.to_categorical(encoded_targets_val, num_classes)

print "\nclasses Val: " + str(le2.classes_)

le3 = LabelEncoder()
le3.fit(refined_targets_test)
encoded_targets_test = le3.transform(refined_targets_test)
categorical_targets_test = keras.utils.to_categorical(encoded_targets_test, num_classes)

print "\nclasses Test: " + str(le3.classes_)




arrRefinedTrain = np.array(refined_train)
arrCategoricalTargetsTrain = np.array(categorical_targets_train)

tmpXTrain = arrRefinedTrain
tmpYTrain = arrCategoricalTargetsTrain

arrRefinedVal = np.array(refined_validation)
arrCategoricalTargetsVal = np.array(categorical_targets_val)

tmpXVal = arrRefinedVal
tmpYVal = arrCategoricalTargetsVal

print 'tmpXTrain.shape: ' + str(tmpXTrain.shape)
print 'tmpYTrain.shape: ' + str(tmpYTrain.shape)
print 'tmpXVal.shape: ' + str(tmpXVal.shape)
print 'tmpYVal.shape: ' + str(tmpYVal.shape)





# ----------------------------- SLIDING WINDOW - SU TRAIN SET E SU VALIDATION SET --------------------




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

print 'x_train.shape after slicing with num_min_frames: ' + str(x_train.shape)
print 'y_train.shape after slicing with num_min_frames: ' + str(y_train.shape)





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

print 'x_val.shape after slicing with num_min_frames: ' + str(x_val.shape)
print 'y_val.shape after slicing with num_min_frames: ' + str(y_val.shape)





# --------------------------------------------------- MODEL TRAINING ----------------------------------------------



model = Sequential()
model.add(LSTM(32, input_shape=(num_min_frames, 24)))  # 17
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


model.fit(x_train, y_train,
          batch_size=128, epochs=200,
          validation_data=(x_val, y_val))


#now = datetime.datetime.now()
#now2 = now.strftime("%Y-%m-%d_%H:%M")
#model.save('/home/gsanesi/PhysioApp/Models2/model_' + str(now2) + '.h5')




# --------------------------------------------------- MODEL TESTING ----------------------------------------------


print "\nVoting Sliding Window on Test Set\n"


arrRefinedTest = np.array(refined_test)
arrCategoricalTargetsTest = np.array(categorical_targets_test)

tmpXTest = arrRefinedTest
tmpYTest = arrCategoricalTargetsTest




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
            print 'QUI'
        else:
            prediction = predicted_classes[0]

        if prediction == np.argmax(tmpYTest[i]): #if np.argmax(tmpYTest[i]) in prediction
            count_correct_classified = count_correct_classified + 1
        else:
            print "predicted: " + str(le.inverse_transform([prediction])) + " with voting " + str(np.amax(voting)) + "/" + str(ii) + " - original: " + str(le.inverse_transform([np.argmax(tmpYTest[i])]))
            print "voting: " + str(voting)
            print "classes: " + str(le.classes_)
            #print "predictions SW Collection: " + str(predsSWCollection)

        pred_confusionMatrix.append(str(le.inverse_transform([prediction])[0]))
        true_confusionMatrix.append(str(le.inverse_transform([np.argmax(tmpYTest[i])])[0]))

print "\ncount_correct_classified: " + str(count_correct_classified)
print 'num_test_examples: ' + str(num_test_examples)
print 'test_accuracy: ' + str(float(count_correct_classified)/num_test_examples)

confusionMatrix_SWVoting = confusion_matrix(true_confusionMatrix, pred_confusionMatrix, labels=["Pick Up", "Throw", "Sitting Down", "Standing Up", "Hand Waving", "Kicking Something", "Hopping", "Jump Up", "Pointing To Something", "Bow"])

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




