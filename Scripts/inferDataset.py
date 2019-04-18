import os
import ffmpy

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


videoFormat = 'avi'

actions = ['A006', 'A007', 'A008', 'A009', 'A023', 'A024', 'A026', 'A027', 'A031', 'A035']


file = "nturgbd_rgb_s017"

count = 0
max = len(os.listdir("/data/datasets/NTURGBD/" + str(file) + "/nturgb+d_rgb"))

for video in os.listdir("/data/datasets/NTURGBD/" + str(file) + "/nturgb+d_rgb"):

    count = count + 1

    if video.endswith(".avi"):

        videoName = os.path.splitext(video)[0]

        print "" + str(count) + "/" + str(max) + " Processing " + str(videoName)

        if 'C001' in videoName:

            checkVideo = False

            for act in actions:
                if (act in videoName) and (videoName + '_framesKeypoints.npy' not in os.listdir("/home/gsanesi/VideoFrames/ResNet50_KeyPointsMask/NTURGBD_C001Keypoints_BestPippo")):
                    checkVideo = True
                    break

            print checkVideo

            if checkVideo is True:

                ffm = ffmpy.FFmpeg(inputs={"/data/datasets/NTURGBD/" + str(file) + "/nturgb+d_rgb/" + str(videoName) + ".avi": None}, outputs={'/home/gsanesi/temporary/scene%05d.png': None})
                ffm.run()

                os.system(
                    'python /home/gsanesi/Desktop/Tesi/densepose/tools/infer_simple.py --cfg /home/gsanesi/Desktop/Tesi/densepose/configs/DensePoseKeyPointsMask_ResNet50_FPN_s1x-e2e.yaml --output-dir /home/gsanesi/VideoFrames/ResNet50_KeyPointsMask/NTURGBD_C001Keypoints_BestPippo --image-ext png --wts https://dl.fbaipublicfiles.com/densepose/DensePoseKeyPointsMask_ResNet50_FPN_s1x-e2e.pkl /home/gsanesi/temporary')


                for image in os.listdir('/home/gsanesi/temporary'):

                    if image.endswith(".png"):

                        os.remove('/home/gsanesi/temporary/' + str(image))


                for fileNPY in os.listdir('/home/gsanesi/VideoFrames/ResNet50_KeyPointsMask/NTURGBD_C001Keypoints_BestPippo'):

                    if fileNPY.endswith(".npy"):

                        if "scene" in str(fileNPY):

                            os.rename('/home/gsanesi/VideoFrames/ResNet50_KeyPointsMask/NTURGBD_C001Keypoints_BestPippo/' + str(fileNPY), '/home/gsanesi/VideoFrames/ResNet50_KeyPointsMask/NTURGBD_C001Keypoints_BestPippo/' + str(videoName) + '_framesKeypoints.npy')

