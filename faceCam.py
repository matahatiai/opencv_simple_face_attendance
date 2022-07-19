import face_recognition
import cv2
import sqlite3
from sqlite3 import Error
import sys
import os
from time import gmtime, strftime
import vlc
import threading
import urllib.parse
import time
import numpy as np
import configparser
from multiprocessing.pool import ThreadPool
import multiprocessing as mp
pool = ThreadPool(processes=4) # 4 workers
print("Total core CPU: %d " % mp.cpu_count())
print('Parent process:', os.getppid())
print('Process id:', os.getpid())

"""
CREATE TABLE "current_face" (
    "id"    INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    "face_index"    INTEGER,
    "user_name" TEXT,
    "create_time"   DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "face.db")
DATENOW = strftime("%Y-%m-%d", gmtime())

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

def get_config(header, index):
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config[header][index]

if(os.path.isfile(DB_PATH) == False):
    print("Database SQLite not found!")
    sys.exit()

def play_sound(name):
    name_query = urllib.parse.quote(name)
    str_query = get_config("APP","tts_url")
    str_query = str_query.replace("{name_query}",name_query)
    p = vlc.MediaPlayer(str_query)
    p.play()

def save_face(face_id, user_name):
    sql = "INSERT INTO current_face (face_id, user_name) VALUES ('"+str(face_id)+"', '"+user_name+"')"
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(sql)
        conn.commit()
    except Error as e:
        print(e)
    finally:
        if conn:
            conn.close()


def check_face(face_id):
    sql = "SELECT COUNT(*)as total FROM current_face \
    WHERE face_id = '"+str(face_id)+"' \
    AND create_time BETWEEN '"+DATENOW+" 00:00' AND '"+DATENOW+" 23:59'"
    conn = None
    result = None
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(sql)
        fetch_data = c.fetchone()
        result = fetch_data[0]
    except Error as e:
        print(e)
    finally:
        if conn:
            conn.close()

    return result

def get_all_face():
    sql = "SELECT * FROM master_face"
    conn = None
    result = None
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(sql)
        result = c.fetchall()
    except Error as e:
        print(e)
    finally:
        if conn:
            conn.close()

    return result

class MyFace(object):
    def __init__(self, id, name, image_name):
        self.id = id
        self.name = name
        self.image_name = image_name

list_faces = []

# Gets all face from db master face
for item in get_all_face():
    list_faces.append(MyFace(item[0],item[1],item[2]))

# Create arrays of known face encodings and their names
known_face_encodings = []

for face_name in list_faces :
    img_load = face_recognition.load_image_file(face_name.image_name)
    img_encoding = face_recognition.face_encodings(img_load)[0]
    known_face_encodings.append(img_encoding)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
face_distances_data = []
process_this_frame = True

# Get a reference to webcam #0 (the default one)
RTSP =  get_config("APP","rtsp_url")
video_capture = cv2.VideoCapture(0)
print("Initialize camera info, Width: %d, Height: %d, FPS: %d" % (video_capture.get(3), video_capture.get(4), video_capture.get(5)))

# run multiprocessing
def vision_processing(frame):
    global process_this_frame, face_locations, face_encodings, face_names, pool, face_distances_data

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        face_distances_data = []

        for face_encoding in face_encodings:

            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # second validate face person
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            face_distances_data.append(face_distances)

            # If matches but compare again
            if True in matches:

                #first_match_index = matches.index(True)
                for i, face_distance in enumerate(face_distances):
                    # with threshold 0.5 is better
                    if(face_distance < 0.5):
                        # get data distances top
                        best_match_index = np.argmin(face_distances)
                        # get name from list by index
                        name = list_faces[best_match_index].name
                        face_id = list_faces[best_match_index].id
                        # check face from SQLite db
                        cf = check_face(face_id)
                        # if check database not found
                        if(cf < 1):
                            threads = []
                            # waiting playsound finished
                            threads.append(threading.Thread(target=play_sound, args=("Selamat datang "+name,)))
                            threads.append(threading.Thread(target=save_face, args=(face_id, name,)))
                            for t in threads :
                                t.start()
                                t.join()

                        # closing iterate
                        break
            # append face for reactangle
            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name, dtc in zip(face_locations, face_names, face_distances_data):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face

        # color red if unknown
        color_frame = (0,0,255)

        # color blue except
        if(name != 'Unknown'):
            color_frame = (255, 102, 51)

        cv2.rectangle(frame, (left, top), (right, bottom), color_frame, 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color_frame, cv2.FILLED)
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        cv2.putText(frame, name + " "+str(dtc), (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    
    return frame

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # reinitialize if connection is broken
    if not(ret):
        st = time.time()
        video_capture = cv2.VideoCapture(RTSP)
        print("Re-Initialize camera info, Width: %d, Height: %d, FPS: %d" % (video_capture.get(3), video_capture.get(4), video_capture.get(5)))
        print("total time lost due to reinitialization : ",time.time()-st)
        continue

    # vision_processing(frame)
    frm = pool.apply_async(vision_processing, (frame,)).get()

    try :
        # Display the resulting image
        cv2.imshow('Employee Attendance', frm)
    except Exception as e :
        pass

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()