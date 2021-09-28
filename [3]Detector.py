#!/usr/bin/env python
# coding: utf-8

# ## ![Logo](http://eswcontest.or.kr/images/openImg.jpg) 
# ## [제19회 임베디드 소프트웨어 경진대회] __양치의 정석__ :: Detector Program

# ## Library Load
# ***

# In[1]:


import cv2
import mediapipe as mp
import numpy as np
import os
from tensorflow.keras.models import load_model
import socket
from datetime import datetime, timedelta
from google.cloud import storage
import json
import pytz


# ## Init & Parameter
# ***

# In[2]:


#Init
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
#model = load_model('Model/L_model.h5')
options = ['none','vertical','horizontal','LU','RU','LD','RD','LU_in','RU_in','LD_in','RD_in']
mouth_landmark = [0,17,57,287,13,14,96,325,138,150,176,152,400,379,367]

#Threshold
ACCURACY_THRESHOLD = 0.8
LANDMARK_NOISE_THRSHOLD = 10
NOISE = np.exp(-10)
MAX_NOISE = np.exp(10)
DIFF_NOISE = 0.5
PREDICT_START_THRESHOLD = 70
BRUSH_START_DISTANCE_THRESHOLD = 0.06
#Socket 
HOST=''
PORT=8999
TIMEOUT = True
#GCP
# GCP Variable
using_bucket_name = "bytmcl"
user_id = "mcl_user"
os.makedirs('GCP/',exist_ok=True)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'GCP/byt_key.json'

if not os.path.isfile('GCP/byt_key.json'):
    print('Please put byt_key.json File in GCP Folder')
    exit()

storage_client = storage.Client()




#Param
seq_length = 60
#left_user = True


# ## Function
# ***

# In[3]:


def recvall(sock, count):
    buf = b''
    
    while count:
        try:
            newbuf = sock.recv(count)
        except:
            print('::Timeout')
            return None
        
        if not newbuf :
            return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def UPLOAD(bucket_name, source_file_name, destination_blob_name):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print("PI {} ---> GCP {} COMPLETE".format(source_file_name , destination_blob_name))
    
def directory_list():   # 디렉토리가 '/'로 끝나는 특징을 사용해 디렉토리 이름만 추출
    play_list_name = member_list_in_bucket(using_bucket_name)
    list_name = []
    for i in range(len(play_list_name)):
        if play_list_name[i][-1:] == '/':
            list_name.append(play_list_name[i][:-1])
    return list_name

def member_list_in_bucket(bucket_name): # 버킷안에 저장된 재생목록 이름들을 불러냄

    blobs = storage_client.list_blobs(bucket_name)
    list_blob = []

    except_str = str(user_id + "/")  # 제외시킬 문자열
    for blob in blobs:
        if blob.name.startswith(except_str):
            blob.name = blob.name.replace(except_str, '')
            if blob.name == '':
                pass
            else:
                list_blob.append(blob.name)
    return list_blob

def time_now(): # 파일이름 저장할 때 사용할 목적의 시간함수
    now = datetime.utcnow()
    UTC = pytz.timezone('UTC')
    now_utc = now.replace(tzinfo=UTC)
    KST = pytz.timezone('Asia/Seoul')
    now_kst = now_utc.astimezone(KST)
    return now_kst

def left_right_hand(index): # 인덱스를 받아서 왼손 or 오른손 여부 반환
    list_member = directory_list()
    member_info = list_member[index]
    hand = member_info[8]
    return int(hand) # 0 -> 오른손잡이 , 1 -> 왼손잡이


# ## Detector
# ***

# In[ ]:


print('::Detector Program Start!!::')
while True:
    
    #Socket Load
    s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print('::Socket Created!!!')

    s.bind((HOST, PORT))
    print('::Socket bind complete!!!')
    s.listen(3)
    print('::Socket now listening')
    conn, addr = s.accept()

    print('::Socket Start!')

    #Variable Init
    Video_index = 0
    Brushing_time = 0
    START_BRUSH = False
    Pred = False
    TIMEOUT = False
    R_hand_past = np.empty([21,3],dtype='float32')
    R_hand_list = np.empty((1,21,3),dtype='float32')
    L_hand_past = np.empty([21,3],dtype='float32')
    L_hand_list = np.empty((1,21,3),dtype='float32')
    mouths_list = np.empty((1,len(mouth_landmark),3),dtype='float32')
    pred_list = [0 for i in range(len(options))]



    with mp_holistic.Holistic(
        min_detection_confidence=0.3,
        min_tracking_confidence=0.5) as holistic:

        while True:
            if not Video_index: #First Frame
                ID = conn.recv(2)
                Video_index += 1
                print('ID : {}'.format(int(ID)))
                left_user = left_right_hand(int(ID))
                if left_user:
                    model = load_model('Model/Left_model.h5')
                    print('Left User')
                else:
                    model = load_model('Model/Right_model.h5')
                    print('Right User')


            #Socket Read & Get Frame
            conn.settimeout(5)
            length = recvall(conn, 16)
            if not length:
                print('::Server Off')
                TIMEOUT = True
                break
            stringData = recvall(conn, int(length))
            if not stringData:
                print('::Server Off')
                TIMEOUT = True
                break
            data = np.frombuffer(stringData, dtype='uint8')
            frame=cv2.imdecode(data, cv2.IMREAD_COLOR)

            if Video_index == 1:
                Frame_X = np.array(frame).shape[1]
                Frame_Y = np.array(frame).shape[0]  




            Video_index += 1

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            GO = False
            MGO = False

            mp_drawing.draw_landmarks(frame,results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame,results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS)

            if not left_user:
                R_hand = np.empty([1,3],dtype='float32')
                if results.right_hand_landmarks:
                    R_hand_center = (results.right_hand_landmarks.landmark[9].x ,results.right_hand_landmarks.landmark[9].y,results.right_hand_landmarks.landmark[9].z)

                    for i,point in enumerate(results.right_hand_landmarks.landmark):
                        a = np.array((point.x,point.y,point.z)).reshape([1,3])
                        R_hand = np.vstack((R_hand,a))
                        if point.x < NOISE or point.y < NOISE:
                            print('::ErrorCode1::',end=' ')
                            break 
                        if point.x > 1 or point.y > 1 or point.z > 1:
                            print('::ErrorCode2::',end=' ')
                            break
                        if not np.exp(point.x) or not np.exp(point.y) or not np.exp(point.z):
                            print('::ErrorCode3::',end=' ')
                            break

                    if len(R_hand) != 1:
                        R_hand = np.delete(R_hand,0,axis=0) 
                        if len(R_hand) != 21 or R_hand.max() > LANDMARK_NOISE_THRSHOLD:
                            print('R_Shape_Error , Shape Len : {}'.format(len(R_hand)))
                            continue
                        a = (R_hand - R_hand_past)*100
                        a = a.reshape([1,21,3])
                        R_hand_list = np.vstack((R_hand_list,a))
                        GO = True
                R_hand_past = R_hand

            else:
                L_hand = np.empty([1,3],dtype='float32')
                if results.left_hand_landmarks:
                    L_hand_center = (results.left_hand_landmarks.landmark[9].x ,results.left_hand_landmarks.landmark[9].y,results.left_hand_landmarks.landmark[9].z)
                    for i,point in enumerate(results.left_hand_landmarks.landmark):
                        a = np.array((point.x,point.y,point.z)).reshape([1,3])
                        L_hand = np.vstack((L_hand,a))
                        if point.x < NOISE or point.y < NOISE:
                            print('::ErrorCode1::',end=' ')
                            break 
                        if point.x > 1 or point.y > 1 or point.z > 1:
                            print('::ErrorCode2::',end=' ')
                            break
                        if not np.exp(point.x) or not np.exp(point.y) or not np.exp(point.z):
                            print('::ErrorCode3::',end=' ')
                            break

                    if len(L_hand) != 1:
                        L_hand = np.delete(L_hand,0,axis=0)
                        if len(L_hand) != 21 or L_hand.max() > LANDMARK_NOISE_THRSHOLD:
                            print('L_Shape_Error , Shape Len : {}'.format(len(L_hand)))
                            continue

                        a = (L_hand - L_hand_past)*100
                        a = a.reshape([1,21,3])
                        L_hand_list = np.append(L_hand_list,a,axis=0)
                        GO = True

                L_hand_past = L_hand
            #endif

            if GO:
                mouth = []
                if results.face_landmarks:
                    mouth.append((results.face_landmarks.landmark[mouth_landmark[0]].x ,results.face_landmarks.landmark[mouth_landmark[0]].y,results.face_landmarks.landmark[mouth_landmark[0]].z))
                    mouth = np.array(mouth)
                    for mouths in mouth_landmark[1:]:
                        a = (results.face_landmarks.landmark[mouths].x,results.face_landmarks.landmark[mouths].y,results.face_landmarks.landmark[mouths].z)
                        a = np.array(a).reshape([1,3])
                        mouth = np.append(mouth,a,axis=0)

                    for i in mouth:
                        cv2.circle(frame,(int(i[0] * Frame_X),int(i[1] * Frame_Y)),2,(0,255,0),-1)

                    if len(mouth) != 1:
                        if (len(mouth) != len(mouth_landmark)) or (mouth.max() > LANDMARK_NOISE_THRSHOLD):
                            print('::Mouth_Shape_Error , Shape Len : {}'.format(len(mouth)))
                            continue

                        if not left_user:
                            mouth = mouth - R_hand_center
                        else :
                            mouth = mouth - L_hand_center

                        mouth = np.where(abs(mouth)>DIFF_NOISE,0,mouth)
                        mouth = np.where(abs(mouth)<NOISE,0,mouth)

                        if not START_BRUSH and (abs(mouth.mean(axis=0)[:2].mean(axis=0)) > BRUSH_START_DISTANCE_THRESHOLD):
                            print('::Waiting for Brushing... Distance : {}'.format(abs(mouth.mean(axis=0)[:2].mean(axis=0)).round(2)))
                            if not left_user:
                                R_hand_list = np.delete(R_hand_list,-1,axis=0)
                            else:
                                L_hand_list = np.delete(L_hand_list,-1,axis=0)

                            continue
                        elif not START_BRUSH and (abs(mouth.mean(axis=0)[:2].mean(axis=0)) < BRUSH_START_DISTANCE_THRESHOLD):
                            print('::Brushing Start !... Distance : {}'.format(abs(mouth.mean(axis=0)[:2].mean(axis=0)).round(2)))
                            START_BRUSH = True


                        a = mouth.reshape([1,len(mouth_landmark),3])
                        mouths_list = np.append(mouths_list,a,axis=0)
                        MGO = True
                        Brushing_time += 1

                if not MGO :
                    if not left_user:
                        R_hand_list = np.delete(R_hand_list,-1,axis=0)
                    else:
                        L_hand_list = np.delete(L_hand_list,-1,axis=0)

            if not left_user:
                Dhand = R_hand_list
            else:
                Dhand = L_hand_list
            #endif

            # Prediction
            if len(Dhand) > PREDICT_START_THRESHOLD:
                input_data = np.hstack((Dhand,mouths_list))
                input_data = np.array(input_data[-seq_length:]).reshape([-1,seq_length,108])
                pred = model.predict(input_data)
                one = pred.argmax()
                two = np.delete(pred,pred.argmax()).argmax()
                mouths_list = np.delete(mouths_list,range(seq_length),axis=0)
                if not left_user:
                    R_hand_list = np.delete(R_hand_list,range(seq_length),axis=0)
                else:
                    L_hand_list = np.delete(L_hand_list,range(seq_length),axis=0)

                if pred[0][one] + pred[0][two] > ACCURACY_THRESHOLD:
                    Pred = True
                    pred_list[one] += int(pred[0][one]*5)/5
                    pred_list[two] += int(pred[0][two]*5)/10
                else:
                    Pred = False


            if Pred:
                cv2.putText(frame,'First  Predict Action : %s' % (options[one]),(20,20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
                cv2.putText(frame,'Second Predict Action : %s' % (options[two]),(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
            cv2.putText(frame,'ID : {}'.format(int(ID)),(Frame_X-120,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
            if not left_user:
                cv2.putText(frame,str(R_hand_list.shape[0]),(20,Frame_Y-20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
            else:
                cv2.putText(frame,str(L_hand_list.shape[0]),(20,Frame_Y-20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                s.close()
                break

    s.close()
    cv2.destroyAllWindows()
    
    if not TIMEOUT:

        # Send to GCP
        list_member = directory_list()
        now_kst = time_now()
        download_data = {  # 여기에 데이터 담아주면됨
            "weakness": { 
                "stop": 1,
                "horizon": 2,
                "vertical": 4,
                "left_up_in": 5,
                "left_up_out": 1200,
                "left_down_in": 500,
                "left_down_out": 7,
                "right_up_in": 8,
                "right_up_out": 30,
                "right_down_in": 20,
                "right_down_out": 3
            },
            "pred_list" : pred_list,
            "score": "100",
            "Brushing_time" : Brushing_time
        }

        with open("GCP/DLdata", 'w', encoding='UTF-8-sig') as make_file:
            make_file.write(json.dumps(download_data, ensure_ascii=False, indent='\t'))

        UPLOAD(using_bucket_name, "GCP/DLdata", user_id + "/" + list_member[int(ID)] + "/" + now_kst.strftime("%Y%m%d%H%M%S"))
        print('GCP Upload Successfully')


# In[ ]:




