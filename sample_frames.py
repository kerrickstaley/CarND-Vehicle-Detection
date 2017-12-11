#!/usr/bin/env python3
import argparse
import os
import cv2
import shutil

FRAME_RATE = 25
PERIOD = 2

parser = argparse.ArgumentParser()
parser.add_argument('vidname')

def main(vidname):
    # assume video is in cwd
    basename = vidname.split('.')[0]
    dirname = f'{basename}_frames'
    shutil.rmtree(dirname, ignore_errors=True)
    os.mkdir(dirname)

    cap = cv2.VideoCapture(vidname)
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if idx % (PERIOD * FRAME_RATE) == 0:
            cv2.imwrite(f'{dirname}/frame_{idx // FRAME_RATE:03}.jpg', frame)
            print('wrote a file!')
        
        idx += 1


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.vidname)
    
