import argparse
import cv2 as cv
from functions import register_face, face_id

parser = argparse.ArgumentParser(description='Face ID')
# option register or detect face
parser.add_argument('--register', action='store_true', help='Register face')
parser.add_argument('--detect', action='store_true', help='Detect face')
# option register has param name 
parser.add_argument('--name', type=str, help='Name of the person')

args = parser.parse_args()

if args.register:
    # open camera
    print('Registering face')
    if args.name is None:
        print('Please provide name of the person')
    else:
        print('Name:', args.name)
        cap = cv.VideoCapture(0)
        register_face(cap,args.name)
elif args.detect:
    print('Detecting face')
    cap = cv.VideoCapture(0)
    face_id(cap)
else:
    print('Please provide option --register or --detect')
