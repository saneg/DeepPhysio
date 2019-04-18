import os
import sys
import numpy as np



min_frames_len = None
count_frames_len = None
n_videos = len(os.listdir("/home/gsanesi/VideoFrames/ResNet50_KeyPointsMask/NTURGBD_C001Keypoints_BestScoreAndArea"))


for file in sorted(os.listdir("/home/gsanesi/VideoFrames/ResNet50_KeyPointsMask/NTURGBD_C001Keypoints_BestScoreAndArea")):

    if file.endswith(".npy"):

        print '\n' + str(file)

        fileNPY = file

        videoKeypoints = np.load('/home/gsanesi/VideoFrames/ResNet50_KeyPointsMask/NTURGBD_C001Keypoints_BestScoreAndArea/' + str(fileNPY) + '')

        if min_frames_len == None:
            min_frames_len = len(videoKeypoints)
        elif len(videoKeypoints) < min_frames_len:
            min_frames_len = len(videoKeypoints)

        if count_frames_len == None:
            count_frames_len = len(videoKeypoints)
        else:
            count_frames_len = count_frames_len + len(videoKeypoints)


        barycenters = []

        for i in range(len(videoKeypoints)):

            sumX = 0
            sumY = 0
            count = 0

            for j in range(len(videoKeypoints[i])):

                if videoKeypoints[i][j][0] is not None and videoKeypoints[i][j][1] is not None:
                    count = count + 1
                    sumX = sumX + videoKeypoints[i][j][0]
                    sumY = sumY + videoKeypoints[i][j][1]

            if count == 0:
                print 'QUI'
                barycenters.append([None, None])
            else:
                barycenters.append([sumX / count, sumY / count])

        print barycenters

        split1 = str(fileNPY).split('_')
        part1FirstVid = split1[0]
        part2FirstVid = split1[2].split('.')[0]


        newVideoKeypoints = []

        for i in range(len(videoKeypoints)):
            nvK = []
            for j in range(len(videoKeypoints[i])):
                if videoKeypoints[i][j][0] is not None and videoKeypoints[i][j][1] is not None:
                    newKey = [(videoKeypoints[i][j][0] - barycenters[i][0]), (videoKeypoints[i][j][1] - barycenters[i][1])]
                    nvK.append(newKey)
                else:
                    nvK.append([None, None])

            newVideoKeypoints.append(nvK)

        np.save('/home/gsanesi/VideoFrames/ResNet50_KeyPointsMask/NTURGBD_C001Dataset_BestScoreAndArea_FramesBar/' + str(part1FirstVid) + '_relBarycenterFramesKeys_' + str(part2FirstVid) + '.npy', np.array(newVideoKeypoints))

        print np.array(videoKeypoints).shape
        print np.array(newVideoKeypoints).shape



print str(min_frames_len)
print str((count_frames_len / n_videos))
