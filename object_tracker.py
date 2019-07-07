from models import *
from utils import *
import cv2
from sort import *
import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from utils.utils import load_classes, non_max_suppression
from PIL import Image
from time import sleep
import numpy
from numpy import ones, vstack
from numpy.linalg import lstsq


class TrackableObject:
    def __init__(self, objectID, centroid):
        # store the object ID, then initialize a list of centroids
        # using the current centroid
        self.objectID = objectID
        self.centroids = [centroid]
        self.lostCount = 0


def detect_image(img):
    # scale and pad image
    ratio = min(img_size / img.size[0], img_size / img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([transforms.Resize((imh, imw)),
                                         transforms.Pad((max(int((imh - imw) / 2), 0), max(int((imw - imh) / 2), 0),
                                                         max(int((imh - imw) / 2), 0), max(int((imw - imh) / 2), 0)),
                                                        (128, 128, 128)),
                                         transforms.ToTensor(),
                                         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = non_max_suppression(detections, 1, conf_thres, nms_thres)
    return detections[0]


def line_eq():
    points = [(1, 5), (3, 4)]
    x_coords, y_coords = zip(*points)
    A = vstack([x_coords, ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords)[0]
    print("Line Solution is y = {m}x + {c}".format(m=m, c=c))


cachedObjects = {}
frame_times = []
videopath = 'video/norm.mp4'
# load weights and set defaults
config_path = 'config/yolov3.cfg'
weights_path = 'config/yolov3.weights'
class_path = 'config/coco.names'
img_size = 416
conf_thres = 0.4
nms_thres = 0.5
line_x1 = 260
line_y1 = 36
line_x2 = 270
line_y2 = 280
skip = 3
# load model and put into eval mode
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
model.cuda()
model.eval()

classes = load_classes(class_path)
Tensor = torch.cuda.FloatTensor

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 0, 128),
          (128, 128, 0), (0, 128, 128)]

videofile = cv2.VideoCapture(videopath)
mot_tracker = Sort()

# Check if camera opened successfully
if (videofile.isOpened() == False):
    print("Error opening video stream or file")

fourcc = cv2.VideoWriter_fourcc(*'XVID')
ret, frame = videofile.read()
vw = frame.shape[1]
vh = frame.shape[0]
print("Video size", vw, vh)
outvideo = cv2.VideoWriter(videopath.replace(".mp4", "-det.mp4"), fourcc, 20.0, (vw, vh))

frames = 0
count_in = 0
count_out = 0
starttime = time.time()

while videofile.isOpened():
    ret, frame = videofile.read()
    if ret == True:
        if frames % skip != 0:
            frames += 1
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pilimg = Image.fromarray(frame)
        detections = detect_image(pilimg)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        img = np.array(pilimg)
        pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
        pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
        unpad_h = img_size - pad_y
        unpad_w = img_size - pad_x

        if detections is not None:
            dets = detections.cpu()
            trackedObjects = mot_tracker.update(dets)
            trackedIds = set(map(lambda dets: dets[4], trackedObjects))
            lostObjIds = set(cachedObjects.keys()).difference(trackedIds)
            print("trackedIds = ", trackedIds)
            print("lostObjIds = ", lostObjIds)
            for lostId in lostObjIds:
                lostObj = cachedObjects[str(lostId)]
                if lostObj.lostCount > 5:
                    print("LOST ID = ", lostId)
                    cachedObjects.pop(str(lostId))
                    #     count_out += 1
                else:
                    cachedObjects[lostId].lostCount += 1
            for x1, y1, x2, y2, obj_id, cls_pred in trackedObjects:
                box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
                box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
                y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
                x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
                color = colors[int(obj_id) % len(colors)]
                cls = classes[int(cls_pred)]
                cv2.rectangle(frame, (x1, y1), (x1 + box_w, y1 + box_h), color, 2)
                cv2.rectangle(frame, (x1, y1 - 35), (x1 + len(cls) * 19 + 80, y1), color, -1)
                cv2.putText(frame, cls + "-" + str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2)

                to = cachedObjects.get(obj_id, None)
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                if to is None:
                    to = TrackableObject(obj_id, center)
                    cachedObjects[str(obj_id)] = to
                else:
                    to.centroids.append(center)
            # cv2.setMouseCallback("Stream", click_and_crop)

        # outvideo.write(frame)
        # cv2.line(frame, (line_x1, line_y1), (line_x2, line_y2), (255, 0, 0), 5)
        cv2.putText(frame, "Out: " + str(int(count_out)), (416, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow('stream', frame)
        # sleep(0.5)
        frames += 1
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

totaltime = time.time() - starttime
print(frames, "frames", frames / totaltime, "fps")
cv2.destroyAllWindows()
outvideo.release()
