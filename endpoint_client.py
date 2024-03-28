import streamlit as st
from streamlit_webrtc import webrtc_streamer
import functions as fn
import qdrant_client as qd
from qdrant_client.models import PointStruct, VectorParams, VectorStruct
import av
import cv2 as cv
import uuid


# connect to qdrant
qd_client = qd.QdrantClient("http://localhost:6333")

try:
# create collection "face_id" if not exists
    qd_client.create_collection("face_id", vectors_config=VectorParams(size=1024, distance="Cosine"))
except:
    pass



st.title("Face ID Client")
st.sidebar.title("Menu")
# button: register face and detect face
menu = st.sidebar.radio("Menu", ["Register Face", "Detect Face"])

def create_transformer(processors):
    def transform(frame: av.VideoFrame):
        img = frame.to_ndarray(format="bgr24")

        for processor in processors:
            if processor is not None:
                img = processor(img)

        return av.VideoFrame.from_ndarray(img, format="bgr24")
    return transform

def register_process(raw_frame:av.VideoFrame):
    frame = raw_frame.to_ndarray(format="bgr24")
    faces = fn.detect_multiple_faces(frame, fn.face_detection)
    print("num of faces: ", len(faces))
    if len(faces) == 1:
        face = faces[0]
        cropped_face = fn.crop_face(frame, face)
        features = fn.extract_features(cropped_face)
        # add vector with metadata name to qdrant
        # features to array
        features = features.numpy()
        qd_client.upsert(collection_name="face_id", points=[PointStruct(vector=features[0], payload={"name": name}, id=uuid.uuid4().hex)])
        # draw rectangle
        x, y, w, h = face
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return av.VideoFrame.from_ndarray(frame, format="bgr24")
    else:
        st.warning("Please provide only one face in the frame")
        return av.VideoFrame.from_ndarray(frame, format="bgr24")

def detect_process(raw_frame:av.VideoFrame):
    frame = raw_frame.to_ndarray(format="bgr24")
    faces = fn.detect_multiple_faces(frame, fn.face_detection)
    if len(faces) > 0:
        for face in faces:
            x, y, w, h = face
            face = fn.crop_face(frame, face)
            if face.shape[0] == 0 or face.shape[1] == 0:
                continue
            features = fn.extract_features(face)
            features = features.numpy()
            # search vector in qdrant
            result = qd_client.search(collection_name="face_id", query_vector=features[0], limit=1, score_threshold=0.9)
            if len(result) > 0:
                name = result[0].payload["name"]
                cv.putText(frame, name, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
    return av.VideoFrame.from_ndarray(frame, format="bgr24")

processor = None

if "processor" not in st.session_state:
    st.session_state.processor = None
else:
    processor = st.session_state.processor

if menu == "Register Face":
    st.subheader("Register Face")
    # open camera
    name = st.text_input("Name")
    if st.button("Start Registering"):
        if name == "":
            st.warning("Please provide name of the person")
        else:
            # register face
            processor = register_process
            st.session_state.processor = processor

if menu == "Detect Face":
    st.subheader("Detect Face")
    if st.button("Start Detecting"):
        processor = detect_process
        st.session_state.processor = processor
    
    

webrtc_streamer(key="streamer", sendback_audio=False, video_frame_callback=processor)
                
