# used to create a proof of concept video of optical flow on road.
import cv2
import os

image_folder = '/road/opf+rgb/2014-06-25-16-45-34_stereo_centre_02/'
video_name = '/road/opf_road.avi'

images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

counter = 0
video = cv2.VideoWriter(video_name, 0, 30, (width,height))
for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))
    counter+=1
    if counter == 360:
        break


cv2.destroyAllWindows()
video.release()