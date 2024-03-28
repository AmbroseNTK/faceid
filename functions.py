import numpy as np
import tensorflow as tf
import mediapipe as mp
import cv2 as cv
import tensorflow_hub as hub
import argparse
import time
import os
import streamlit as st


mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection()


conv_layer = hub.KerasLayer("https://www.kaggle.com/models/google/mobilenet-v3/frameworks/TensorFlow2/variations/small-100-224-feature-vector/versions/1",
                   trainable=False)

# show camera


def crop_face(frame, face):
    x, y, w, h = face
    return frame[y:y+h, x:x+w]

def detect_face(frame, face_detection):
    # using mediapipe to detect face
    
    results = face_detection.process(frame)
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            return x, y, w, h
    return None

def detect_multiple_faces(frame, face_detection):
    results = face_detection.process(frame)
    faces = []
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            faces.append((x, y, w, h))
    return faces

def preprocess_image(image):
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    return image

def extract_features(face):
    # resize image to 224x224
    face = tf.image.resize(face, (224, 224))
    # convert image to float32
    face = tf.cast(face, tf.float32)
    # normalize image
    face = face / 255.0
    # expand dimension
    face = tf.expand_dims(face, axis=0)
    # extract features
    features = conv_layer(face)
    return features

def register_face(cap,name):
    num_of_faces = 10 
    data = []
    while cap.isOpened() and num_of_faces > 0:
        ret, frame = cap.read()
        if not ret:
            continue
        face = detect_face(frame, face_detection)
        if face is not None:
            x, y, w, h = face
            face = crop_face(frame, face)
            features = extract_features(face)
            data.append(features)
            num_of_faces -= 1
            # draw rectangle
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv.imshow('Camera', frame)
        time.sleep(1)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    np.save(f'./data/{name}.npy', data)
    cap.release()
    print(data)

def load_data():
    data = []
    for file in os.listdir('./data'):
        if file.endswith('.npy'):
            features = np.load(f'./data/{file}')
            name = file.split('.')[0]
            for feature in features:
                data.append((name, feature))
    return data

def cosine_similarity(a, b):
    # a and b is (1,1024)
    a = a.numpy()
    # b is ndarray
    dot = np.dot(a, b.T)
    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    return dot / (norma * normb)


def face_id(cap):
    # load data
    data = load_data()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
        faces = detect_multiple_faces(frame, face_detection)
        for face in faces:
       
            if face is not None:
                x, y, w, h = face
                face = crop_face(frame, face)
                if face.shape[0] == 0 or face.shape[1] == 0:
                    continue
                extracted_feature = extract_features(face)
            
                for name, feature in data:
                    similarity = cosine_similarity(extracted_feature, feature)
                    print(name, similarity)
                    if similarity > 0.9:
                        cv.putText(frame, name, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        break
                cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv.imshow('Camera', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            break
    
