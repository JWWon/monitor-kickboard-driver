import argparse
import os
import sys
from pathlib import Path
import datetime

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import psutil

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, \
    increment_path, non_max_suppression, print_args, save_one_box, scale_coords, set_logging, \
    strip_optimizer, xyxy2xywh
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync

from PyQt5.QtWidgets import *
from PyQt5 import uic, QtGui, QtCore

form_class = uic.loadUiType("app.ui")[0]

class App_WindowClass(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        check_requirements(exclude=('tensorboard', 'thop'))
        
        self.actionStart_Detecting.triggered.connect(self.run)
    
    def run(self):
        SOURCE = '0' #webcam
        WEIGHTS = 'best.pt'
        IMG_SIZE = 640
        DEVICE = ''
        AUGMENT = False
        CONF_THRES = 0.25
        IOU_THRES = 0.45
        CLASSES = None
        AGNOSTIC_NMS = False
        
        source, weights, imgsz = SOURCE, WEIGHTS, IMG_SIZE

        # Initialize
        device = select_device(DEVICE)
        half = device.type != 'cpu'  # half precision only supported on CUDA
        print('device:', device)

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if half:
            model.half()  # to FP16

        # Get names
        names = model.module.names if hasattr(model, 'module') else model.names
        print('class names = ', names)
        
        # webcam dataloader
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(SOURCE, img_size=imgsz, stride=stride, auto=True)
        bs = len(dataset)  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference once
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
        dt, seen = [0.0, 0.0, 0.0], 0
        
        # Run inference cycle
        for path, img, im0s, vid_cap in dataset:
            t1 = time_sync()
            
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float() # uint8 to fp16/32
            img = img / 255.0  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1
            
            # Inference
            visualize = False
            pred = model(img, augment=AUGMENT, visualize=visualize)[0]
            
            t3 = time_sync()
            dt[1] += t3 - t2
            
            # NMS
            pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, CLASSES, AGNOSTIC_NMS, max_det=20) # detect up to 20
            dt[2] += time_sync() - t3
            
            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count # batch_size >= 1
                
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0
                annotator = Annotator(im0, line_width=3, example=str(names))
                
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        # Add bbox to image
                        c = int(cls)  # integer class
                        label = f'{names[c]} {conf:.2f}'
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        
            logfile = open("detection_log_GUI_pt.txt", "a")
            
            # Print time and detection results (inference-only)
            time_now = datetime.datetime.now()
            print(time_now.strftime('%Y-%m-%d %H:%M:%S'))
            self.label_datetime.setText(time_now.strftime('%Y-%m-%d %H:%M:%S'))
            print(f'{s}detected. (time to infer : {t3 - t2:.3f}s)')
            self.label_time_to_infer.setText(f'{t3 - t2:.3f}s')
            self.textBrowser_detected_classes.setText(f'{s}')
            
            logfile.write(time_now.strftime('%Y-%m-%d %H:%M:%S') + f'{s}detected. (time to infer : {t3 - t2:.3f}s)' + '\n')
            
            split=s.split()
            
            # check helmet and fellows and warn if detected
            warnings = str()
            if ('Helmet,' not in split or 'Helmets,' not in split) and ('Persons,' in split or 'Person,' in split):
                print("WARNING : no helmet detected.")
                warnings += 'No helmet detected.'
                print('\a')
                
            if 'Persons,' in split or 'Helmets,' in split:
                print("WARNING : fellow rider detected.")
                warnings += ' fellow rider detected.'
                print('\a')
                    
            self.statusbar.showMessage(warnings)
            self._print_usage_of_cpu_and_memory(logfile)
            logfile.close()
            
            # Stream results
            im0 = annotator.result()
            im0 = QtGui.QImage(im0.data, im0.shape[1], im0.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
            pixmap = QtGui.QPixmap.fromImage(im0)
            self.label_image.setPixmap(pixmap)
            self.label_image.resize(471, 401)
            cv2.waitKey(1)  # 1 millisecond
            
            
    def _print_usage_of_cpu_and_memory(self, logfile):
        pid = os.getpid()
        cur_process = psutil.Process(pid)
        
        cpu_usage = psutil.cpu_percent()
        memory_usage  = cur_process.memory_info()[0] / 2.**20

        self.label_cpu_usage.setText(str(cpu_usage) + "%")
        self.label_memory_usage.setText(str(memory_usage) + "MB")
        
        logfile.write(str(cpu_usage) + ' ' + str(memory_usage) + ' ' + '\n')


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    appWindow = App_WindowClass()
    appWindow.show()
    
    app.exec_()
