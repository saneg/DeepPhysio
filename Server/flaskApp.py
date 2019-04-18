from __future__ import print_function
from flask import Flask
from flask import request
from flask import json
from flask import send_from_directory
from flask_socketio import SocketIO
from flask_socketio import emit

import sys
import numpy as np
import os
import ffmpy
import keras
from keras import backend as K
import threading

from sklearn.preprocessing import LabelEncoder

from keras.models import load_model

from keras.backend.tensorflow_backend import set_session

import tensorflow as tf

app = Flask(__name__)
socketio = SocketIO(app, async_handlers=True)


users = [
    {
        'userID': 'user_',
        'password': 'deepPhysio',
        'socketID': None
    }
]


activities = [
    {
        'userID': 'user_',
        'activity': 'Pick Up',
        'done': False,
        'videoPath': None,
        'prediction': None,
        'correctnessRatio': None,
        'acquisitionDate': None
    },
    {
        'userID': 'user_',
        'activity': 'Throw',
        'done': False,
        'videoPath': None,
        'prediction': None,
        'correctnessRatio': None,
        'acquisitionDate': None
    },
    {
        'userID': 'user_',
        'activity': 'Sitting Down',
        'done': False,
        'videoPath': None,
        'prediction': None,
        'correctnessRatio': None,
        'acquisitionDate': None
    },
    {
        'userID': 'user_',
        'activity': 'Standing Up',
        'done': False,
        'videoPath': None,
        'prediction': None,
        'correctnessRatio': None,
        'acquisitionDate': None
    },
    {
        'userID': 'user_',
        'activity': 'Hand Waving',
        'done': False,
        'videoPath': None,
        'prediction': None,
        'correctnessRatio': None,
        'acquisitionDate': None
    },
    {
        'userID': 'user_',
        'activity': 'Kicking Something',
        'done': False,
        'videoPath': None,
        'prediction': None,
        'correctnessRatio': None,
        'acquisitionDate': None
    },
    {
        'userID': 'user_',
        'activity': 'Hopping',
        'done': False,
        'videoPath': None,
        'prediction': None,
        'correctnessRatio': None,
        'acquisitionDate': None
    },
    {
        'userID': 'user_',
        'activity': 'Jump Up',
        'done': False,
        'videoPath': None,
        'prediction': None,
        'correctnessRatio': None,
        'acquisitionDate': None
    },
    {
        'userID': 'user_',
        'activity': 'Pointing To Something',
        'done': False,
        'videoPath': None,
        'prediction': None,
        'correctnessRatio': None,
        'acquisitionDate': None
    },
    {
        'userID': 'user_',
        'activity': 'Bow',
        'done': False,
        'videoPath': None,
        'prediction': None,
        'correctnessRatio': None,
        'acquisitionDate': None
    }
]



chat = [
    {
        'senderID': "physio1",
        'receiverID': "user_",
        'message': "Hi",
        'sendDate': "2019-02-11 15:22:43"
    },
    {
        'senderID': "user_",
        'receiverID': "physio1",
        'message': "Hi",
        'sendDate': "2019-02-11 16:30:43"
    }
]


num_min_frames = 29

rUserId = None
rActivity = None
rDate = None
elaborateOrder = []

sem = threading.Semaphore()


@app.route('/', methods=['POST'])
def processVideo():

    print(str(request.form['date']), file=sys.stderr)
    print(str(request.form['user']), file=sys.stderr)
    print(str(request.form['activity']), file=sys.stderr)

    if request.method == 'POST':
        video = request.files['the_video']
        if not os.path.exists('/home/gsanesi/PhysioApp/Videos/' + str(request.form['user']) + ''):
            os.makedirs('/home/gsanesi/PhysioApp/Videos/' + str(request.form['user']) + '')

        video.save('/home/gsanesi/PhysioApp/Videos/' + str(request.form['user']) + '/' + str(request.form['activity']).replace(" ", "_") + '_' + str(request.form['user']) + '_' + str(request.form['date']).replace(" ", "_") + '.mp4')

        for elem in activities:
            if (elem['userID'] == str(request.form['user'])) and (elem['activity'] == str(request.form['activity'])):
                elem['done'] = True
                elem['videoPath'] = '/' + str(request.form['user']) + '/' + str(request.form['activity']).replace(" ", "_") + '_' + str(request.form['user']) + '_' + str(request.form['date']).replace(" ", "_") + '.mp4'
                elem['prediction'] = None
                elem['acquisitionDate'] = request.form['date']
                break

        data = {
            'userID': request.form['user'],
            'activityName': request.form['activity'],
            'videoDate': request.form['date']
        }

        response = app.response_class(
            response=json.dumps(data),
            status=200,
            mimetype='application/json'
        )

        return response




@app.route('/<path:subpath>/<vid_name>')
def serve_video(subpath, vid_name):
    return send_from_directory("/home/gsanesi/PhysioApp/Videos/" + subpath + "/", vid_name, as_attachment=True)


@socketio.on('elaborateVideo')
def elaborateVideo_func(requestUserID, requestActivity, requestDate):

    global rUserId
    global rActivity
    global rDate

    elaborateOrder.append({
        'userID': requestUserID,
        'activity': requestActivity,
        'date': requestDate
    })

    sem.acquire()

    elemPopped = elaborateOrder.pop(0)


    rUserId = elemPopped['userID']
    rActivity = elemPopped['activity']
    rDate = elemPopped['date']



    socketID = None
    for elem in users:
        if (elem['userID'] == rUserId):
            socketID = elem['socketID']

    ffm = ffmpy.FFmpeg(
        inputs={'/home/gsanesi/PhysioApp/Videos/' + str(rUserId) + '/' + str(rActivity).replace(" ", "_") + '_' + str(rUserId) + '_' + str(rDate).replace(" ", "_") + '.mp4': None},
        outputs={'/home/gsanesi/temporary/scene%05d.png': None})
    ffm.run()

    if not os.path.exists('/home/gsanesi/PhysioApp/ExtractedData/' + str(rUserId) + ''):
        os.makedirs('/home/gsanesi/PhysioApp/ExtractedData/' + str(rUserId) + '')

    if not os.path.exists('/home/gsanesi/PhysioApp/ExtractedData/' + str(rUserId) + '/keypoints_images'):
        os.makedirs('/home/gsanesi/PhysioApp/ExtractedData/' + str(rUserId) + '/keypoints_images')

    if not os.path.exists('/home/gsanesi/PhysioApp/VideoWKeypoints/' + str(rUserId) + '/' + str(rActivity).replace(" ","_") + '_' + str(rUserId) + '_' + str(rDate).replace(" ", "_") + ''):
        os.makedirs('/home/gsanesi/PhysioApp/VideoWKeypoints/' + str(rUserId) + '/' + str(rActivity).replace(" ","_") + '_' + str(rUserId) + '_' + str(rDate).replace(" ", "_") + '')

    os.system(
        'python /home/gsanesi/Desktop/Tesi/densepose/tools/infer_simple.py --cfg /home/gsanesi/Desktop/Tesi/densepose/configs/DensePoseKeyPointsMask_ResNet50_FPN_s1x-e2e.yaml --output-dir /home/gsanesi/PhysioApp/ExtractedData/' + str(rUserId) + ' --image-ext png --wts https://dl.fbaipublicfiles.com/densepose/DensePoseKeyPointsMask_ResNet50_FPN_s1x-e2e.pkl /home/gsanesi/temporary')


    for image in os.listdir('/home/gsanesi/temporary'):
        if image.endswith(".png"):
            os.remove('/home/gsanesi/temporary/' + str(image))

    for image in os.listdir('/home/gsanesi/PhysioApp/ExtractedData/' + str(rUserId) + '/keypoints_images'):
        if image.endswith(".png"):
            os.system('cp /home/gsanesi/PhysioApp/ExtractedData/' + str(rUserId) + '/keypoints_images/' + str(image) + ' /home/gsanesi/PhysioApp/VideoWKeypoints/' + str(rUserId) + '/' + str(rActivity).replace(" ", "_") + '_' + str(rUserId) + '_' + str(rDate).replace(" ","_") + '/' + str(image))
            os.remove('/home/gsanesi/PhysioApp/ExtractedData/' + str(rUserId) + '/keypoints_images/' + str(image))

    for fileNPY in os.listdir('/home/gsanesi/PhysioApp/ExtractedData/' + str(rUserId) + ''):
        if fileNPY.endswith(".npy"):
            if "scene" in str(fileNPY):
                os.rename(
                     '/home/gsanesi/PhysioApp/ExtractedData/' + str(rUserId) + '/' + str(fileNPY),
                    '/home/gsanesi/PhysioApp/ExtractedData/' + str(rUserId) + '/' + str(rActivity).replace(" ", "_") + '_' + str(rUserId) + '_' + str(rDate).replace(" ", "_") + '.npy')

    vKeypoints = np.load(
        '/home/gsanesi/PhysioApp/ExtractedData/' + str(rUserId) + '/' + str(rActivity).replace(" ", "_") + '_' + str(rUserId) + '_' + str(rDate).replace(" ", "_") + '.npy')

    barycenters = []

    for i in range(len(vKeypoints)):

        sumX = 0
        sumY = 0
        count = 0

        for j in range(len(vKeypoints[i])):

            if vKeypoints[i][j][0] is not None and vKeypoints[i][j][1] is not None:
                count = count + 1
                sumX = sumX + vKeypoints[i][j][0]
                sumY = sumY + vKeypoints[i][j][1]

        if count == 0:
            barycenters.append([None, None])
        else:
            barycenters.append([sumX / count, sumY / count])

    barX = 0
    barY = 0
    countBar = 0
    for i in range(len(barycenters)):

        if barycenters[i][0] is not None and barycenters[i][1] is not None:
            countBar = countBar + 1
            barX = barX + barycenters[i][0]
            barY = barY + barycenters[i][1]

    allFrameBarycenter = [barX / countBar, barY / countBar]

    newVKeypoints = []

    for i in range(len(vKeypoints)):
        nvidK = []
        for j in range(len(vKeypoints[i])):
            if vKeypoints[i][j][0] is not None and vKeypoints[i][j][1] is not None:
                newKey = [(vKeypoints[i][j][0] - allFrameBarycenter[0]), (vKeypoints[i][j][1] - allFrameBarycenter[1])]
                nvidK.append(newKey)
            else:
                nvidK.append([None, None])

        newVKeypoints.append(nvidK)

    if not os.path.exists('/home/gsanesi/PhysioApp/BarycenteredData/' + str(rUserId) + ''):
        os.makedirs('/home/gsanesi/PhysioApp/BarycenteredData/' + str(rUserId) + '')

    np.save('/home/gsanesi/PhysioApp/BarycenteredData/' + str(rUserId) + '/' + str(rActivity).replace(" ", "_") + '_' + str(rUserId) + '_' + str(rDate).replace(" ", "_") + '_barycentered.npy', np.array(newVKeypoints))

    videoKeypoints = np.load('/home/gsanesi/PhysioApp/BarycenteredData/' + str(rUserId) + '/' + str(rActivity).replace(" ", "_") + '_' + str(rUserId) + '_' + str(rDate).replace(" ", "_") + '_barycentered.npy')

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

        arrTestVid = np.array(reshapedNewVideoKeypoints)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        sess = tf.Session(config=config)
        set_session(sess)

        model = load_model('/home/gsanesi/PhysioApp/Models/model_2019-02-21_13:19.h5')

        classes = ['Bow', 'Hand Waving', 'Hopping', 'Jump Up', 'Kicking Something', 'Pick Up', 'Pointing To Something', 'Sitting Down', 'Standing Up', 'Throw']

        le = LabelEncoder()
        le.fit(classes)
        encoded_target = le.transform([str(rActivity)])
        categorical_target = keras.utils.to_categorical(encoded_target, 10)

        arrCategoricalTarget= np.array(categorical_target)

        predictionRatio = None

        if len(arrTestVid) == num_min_frames:

            x_test = arrTestVid.reshape(1, num_min_frames, 24)

            results = model.predict(x_test, batch_size=1)

            max_value = np.amax(results)
            indexOfMax = np.argmax(results)
            if indexOfMax == np.argmax(arrCategoricalTarget[0]):
                predictionBool = True
                actionPredicted = str(le.inverse_transform([indexOfMax])[0])
                predictionRatio = max_value
            else:
                predictionBool = False
                actionPredicted = str(le.inverse_transform([indexOfMax])[0])

        else:
            ii = 0
            voting = np.zeros((10,), dtype=int)
            predsSWCollection = []

            sum_prediction_proba = None

            while ((num_min_frames + ii) <= len(arrTestVid)):
                x_test = arrTestVid[(0 + ii):(num_min_frames + ii)].reshape(1, num_min_frames, 24)

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

            predicted_classes = [indice for indice in range(len(voting)) if voting[indice] == np.amax(voting)]

            if len(predicted_classes) > 1:

                avarage_prediction_proba = [round(float(sum_prediction_proba[0][ind]) / ii, 2) for ind in range(len(sum_prediction_proba[0]))]

                limitedAvaragePredProba = [avarage_prediction_proba[index] for index in predicted_classes]

                prediction = predicted_classes[np.argmax(limitedAvaragePredProba)]
            else:
                prediction = predicted_classes[0]

            if prediction == np.argmax(arrCategoricalTarget[0]):

                sorted_voting = list(voting)

                sorted_voting.sort(reverse=True)

                predictionBool = True
                predictionRatio = float(sorted_voting[1])/sorted_voting[0]
                actionPredicted = str(le.inverse_transform([prediction])[0])

            else:

                predictionBool = False
                actionPredicted = str(le.inverse_transform([prediction])[0])

        emit('updateActivity', (str(rActivity), predictionBool), room=socketID)

        for elem in activities:
            if (elem['userID'] == str(rUserId)) and (elem['activity'] == str(rActivity)):
                elem['prediction'] = actionPredicted
                elem['correctnessRatio'] = predictionRatio

        K.clear_session()

        sem.release()

        rUserId = None
        rActivity = None
        rDate = None

    else:

        emit('updateActivitiesSinceAlert', str(rActivity), room=socketID)

        emit('alertVideoNotCorrect', str(rActivity), room=socketID)

        for elem in activities:
            if (elem['userID'] == str(rUserId)) and (elem['activity'] == str(rActivity)):
                elem['done'] = False
                elem['prediction'] = None
                elem['acquisitionDate'] = None

        sem.release()

        rUserId = None
        rActivity = None
        rDate = None





@socketio.on('logIn')
def logIn_func(userID, password):

    response = False
    for elem in users:
        if (elem['userID'] == userID) and (elem['password'] == password):
            elem['socketID'] = request.sid
            response = True
            break

    emit('logIn_response', (response, userID, password))


@socketio.on('update_userSocketID')
def updateSocketID_func(userID, password):

    for elem in users:
        if (elem['userID'] == userID) and (elem['password'] == password):
            elem['socketID'] = request.sid
            break






@socketio.on('openActivitiesPage')
def openActPage(userID):

    selected_activities = []

    for elem in activities:
        if (elem['userID'] == userID) and (elem['done'] is False):
            selected_activities.append(elem)

    name_selected_activities = [element['activity'] for element in selected_activities]
    sortedInds_name_sel = np.argsort(name_selected_activities)

    user_activities = []

    for i in sortedInds_name_sel:
        user_activities.append(selected_activities[i]['activity'])

    emit('openActivitiesPage_response', user_activities)





@socketio.on('go2ListPage')
def openListPage(userID):

    selected_activities = []

    for elem in activities:
        if (elem['userID'] == userID) and (elem['done'] is False):
            selected_activities.append(elem)

    name_selected_activities = [element['activity'] for element in selected_activities]
    sortedInds_name_sel = np.argsort(name_selected_activities)

    user_activities = []

    for i in sortedInds_name_sel:
        user_activities.append(selected_activities[i]['activity'])

    emit('go2ListPage_response', user_activities)


@socketio.on('go2DonePage')
def openDonePage(userID):


    done_pred_activities = []
    done_unpred_activities = []

    for elem in activities:
        if (elem['userID'] == userID) and (elem['done'] is True):
            if (elem['prediction'] is None):
                done_unpred_activities.append(elem)
            else:
                done_pred_activities.append(elem)

    date_done_pred = [element['acquisitionDate'] for element in done_pred_activities]
    date_done_unpred = [element['acquisitionDate'] for element in done_unpred_activities]

    sortedInds_done_pred = np.argsort(date_done_pred)
    sortedInds_done_unpred = np.argsort(date_done_unpred)

    user_activities = []
    status_activities = []

    for i in sortedInds_done_pred:
        user_activities.append(done_pred_activities[i]['activity'])
        if (done_pred_activities[i]['activity'].lower() == done_pred_activities[i]['prediction'].lower()):
            status_activities.append(True)
        else:
            status_activities.append(False)

    for i in sortedInds_done_unpred:
        user_activities.append(done_unpred_activities[i]['activity'])
        status_activities.append(None)

    emit('go2DonePage_response', (user_activities, status_activities))






@socketio.on('go2VideoReviewPage')
def openVideoReviewPage(userID, activity):

    videoResults = None
    for elem in activities:
        if (elem['userID'] == userID) and (elem['activity'] == activity):
            if (elem['activity'].lower() == elem['prediction'].lower()):
                videoResults = {
                    'activity': elem['activity'],
                    'videoPath': elem['videoPath'],
                    'prediction': elem['prediction'].upper(),
                    'ratio': elem['correctnessRatio'],
                    'check': True
                }
            else:
                videoResults = {
                    'activity': elem['activity'],
                    'videoPath': elem['videoPath'],
                    'prediction': elem['prediction'].upper(),
                    'ratio': elem['correctnessRatio'],
                    'check': False
                }

    emit('go2VideoReviewPage_response', videoResults)








@socketio.on('go2ChatPage_actPage')
def openChatPage_actPage(userID):

    messages=[]
    for elem in chat:
        if (elem['senderID'] == userID and elem['receiverID'] == 'physio1') or (elem['senderID'] == 'physio1' and elem['receiverID'] == userID):
            messages.append(elem)

    emit('go2ChatPage_actPage_response', messages)

@socketio.on('go2ChatPage_videoRPage')
def openChatPage_videoRPage(userID):

    messages=[]
    for elem in chat:
        if (elem['senderID'] == userID and elem['receiverID'] == 'physio1') or (elem['senderID'] == 'physio1' and elem['receiverID'] == userID):
            messages.append(elem)

    emit('go2ChatPage_videoRPage_response', messages)


@socketio.on('sendMessage')
def sendMessageFunc(userID, message, date):

    newMessage = {
        'senderID': userID,
        'receiverID': "physio1",
        'message': message,
        'sendDate': date,
    }
    chat.append(newMessage)

    senderSocket = None
    receiverSocket = None

    for elem in users:
        if elem['userID'] == userID:
            senderSocket = elem['socketID']

    #for elem in users:
    #    if elem['userID'] == receiverID:
    #        senderSocket = elem['socketID']

    print(str(senderSocket), file=sys.stderr)

    emit('newMessage', newMessage, room=senderSocket)
    #emit('newMessage', newMessage, room=receiverSocket)




if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0')