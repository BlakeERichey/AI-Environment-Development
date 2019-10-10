#convert video to images

video_name = 'bottle.mp4'
dest = './images/'

import cv2
vidcap = cv2.VideoCapture(video_name)
def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite(dest+"image"+str(count)+".jpg", image)     # save frame as JPG file
    return hasFrames


sec = 0
frameRate = 0.05 #//it will capture image in each 0.5 second
count=0
success = getFrame(sec)
count+=1
while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)