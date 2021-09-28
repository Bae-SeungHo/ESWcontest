#!/usr/bin/env python
# coding: utf-8

# ## ![Logo](http://eswcontest.or.kr/images/openImg.jpg) 
# ## [제19회 임베디드 소프트웨어 경진대회] __양치의 정석__ :: Dataset Collecting Program

# ## Library Load
# ***

# In[ ]:


import cv2
import mediapipe as mp
import numpy as np
import os


# ## INIT & Parameter
# ***

# In[ ]:


### MediaPipe Load
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

os.makedirs('Datasets/',exist_ok=True)
Hand = ['Left','Right']
Options = ['None','Vertical','Horizontal','LU','RU','LD','RD','LU_in','RU_in','LD_in','RD_in']
mouth_landmark = [0,17,57,287,13,14,96,325,138,150,176,152,400,379,367]


# In[ ]:


print('Dataset Collector Program Start!')
cam = cv2.VideoCapture(0)
while True:
    status,frame = cam.read()
    if status:
        print('Camera on')
        break
####
Frame_X = np.array(frame).shape[1]
Frame_Y = np.array(frame).shape[0]


# In[ ]:


LANDMARK_NOISE_THRSHOLD = 10
STUDY_FRAME = 1000
NOISE = np.exp(-10)
MAX_NOISE = np.exp(10)
DIFF_NOISE = 0.5


# ## Dataset Collector
# ***

# In[ ]:


print('::Dataset Collecting Procedure ::')
for hand in Hand:
    print('%-5s %10s'%(hand,Options[0]),end='')
    for option in Options[1:]:
        print('->',end='')
        print('%s' % (option),end='')
    print()
    
cv2.waitKey(5000)

with mp_holistic.Holistic(
    min_detection_confidence=0.3,
    min_tracking_confidence=0.5) as holistic:
    
    for hand in Hand:
        for Option in Options:
            
            if os.path.isfile('Datasets/%s_%s.npy' % (hand,Option)):
                print('::{} {} Already Exist.... Skip'.format(hand,Option))
                continue
            else:   
                print('::{} {} Ready.... '.format(hand,Option),end='')
                cv2.waitKey(5000)
                print('Start')
            
            Video_index = 0
            R_hand_past = np.empty([21,3],dtype='float32')
            R_hand_list = np.empty((1,21,3),dtype='float32')
            L_hand_past = np.empty([21,3],dtype='float32')
            L_hand_list = np.empty((1,21,3),dtype='float32')
            mouths_list = np.empty((1,len(mouth_landmark),3),dtype='float32')
            R_hand_center = 0
            L_hand_center = 0
            
            while Video_index <= STUDY_FRAME:
    
                status, frame = cam.read()
                if not status:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                LGO = False
                RGO = False
                MGO = False

                if hand == 'Right':
                # Draw Right Hand
                    mp_drawing.draw_landmarks(frame,results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS)
                else:
                # Draw Left Hand
                    mp_drawing.draw_landmarks(frame,results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS)
                
                
                if hand == 'Right':
                    
                    R_hand = np.empty([1,3],dtype='float32')
                    if results.right_hand_landmarks:
                        R_hand_center = (results.right_hand_landmarks.landmark[9].x ,results.right_hand_landmarks.landmark[9].y,results.right_hand_landmarks.landmark[9].z) # 

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
                            RGO = True

                    R_hand_past = R_hand
                    
                else: # Left Hand 
                    
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
                            LGO = True

                    L_hand_past = L_hand

                    
                if (hand == 'Right' and RGO) or (hand == 'Left' and LGO): 
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
                            if hand == 'Right':
                                mouth = mouth - R_hand_center
                            else :
                                mouth = mouth - L_hand_center
                                
                            mouth = np.where(abs(mouth)>DIFF_NOISE,0,mouth)
                            mouth = np.where(abs(mouth)<NOISE,0,mouth)
                            
                            a = mouth.reshape([1,len(mouth_landmark),3])
                            mouths_list = np.append(mouths_list,a,axis=0)
                            
                            MGO = True
                    
                    
                    if not MGO :
                        if hand == 'Right':
                            R_hand_list = np.delete(R_hand_list,-1,axis=0)
                        else:
                            L_hand_list = np.delete(L_hand_list,-1,axis=0)
                    

                Video_index += 1
                if hand == 'Right':
                    cv2.putText(frame,'Dataset Count : %d' % (R_hand_list.shape[0]),(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0))
                else:   
                    cv2.putText(frame,'Dataset Count : %d' % (L_hand_list.shape[0]),(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0))
                #cv2.putText(frame,str(mouths_list.shape[0]),(40,240),cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0))  
                cv2.imshow('frame', frame)
                
                if cv2.waitKey(1) == ord('q'):
                    break
            ### for Each Option video
            
            mouths_list = np.delete(mouths_list,0,axis=0)
            if hand == 'Right':
                Dhand = R_hand_list
            else:
                Dhand = L_hand_list
            try:
                Dhand = np.delete(Dhand,(0,1),axis=0)
                Dhand = np.hstack((Dhand,mouths_list))
                Dhand = np.delete(Dhand,np.where(Dhand>DIFF_NOISE*100),axis=0)

            except:
                print("::Couldn't Find Hand and Face... Skip")
                continue
            
            np.save('Datasets/%s_%s' % (hand,Option),Dhand)
            print('::{} {} Record Ended....'.format(hand,Option))
            
            cv2.waitKey(3000)
                
### End Collecting Datasets
cam.release()
cv2.destroyAllWindows()
print("::Program End!")

