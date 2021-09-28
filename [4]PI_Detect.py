import RPi.GPIO as gpio
import time
from multiprocessing import Process
import socket, cv2
import numpy as np
import struct
import os
import pigpio
pi = pigpio.pi()
pi.set_servo_pulsewidth(27, 0)
pi.set_servo_pulsewidth(27, 500)
##os.system("sudo pigpiod")


gpio.setmode(gpio.BCM)

first_trig = 17 # 23
first_echo = 18 # 24

second_trig = 23
second_echo = 24

gpio.setup(second_trig, gpio.OUT)
gpio.setup(first_trig, gpio.OUT)

gpio.setup(first_echo, gpio.IN)
gpio.setup(second_echo, gpio.IN)
FIRST_COMM = False
SECOND_COMM = False
CONNECT_FLAG = False
FIRST_CONNECT = False
SECOND_CONNECT = False

HOST = '165.229.187.226'
PORT = 8999

CAMERA = 0
cam = 0
s = 0

def camera():
    print("\n @@@@@@ Start Camera Process @@@@@")
    global cam , end_status
    cam = cv2.VideoCapture(0)
    ## 이미지 속성 변경 3 = width, 4 = height
    cam.set(3, 640);
    cam.set(4, 480);
     
    ## 0~100에서 90의 이미지 품질로 설정 (default = 95)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 99]
    
    while True:
        global s
        # 비디오의 한    프레임씩 읽는다.
        # 제대로 읽으면 ret = True, 실패면 ret = False, frame에는 읽은 프레임
        ret, frame = cam.read()
        # cv2. imencode(ext, img [, params])
        # encode_param의 형식으로 frame을 jpg로 이미지를 인코딩한다.
        result, frame = cv2.imencode('.jpg', frame, encode_param)


        # frame을 String 형태로 변환
        data = np.array(frame)
        stringData = data.tostring()
        message_size = struct.pack("L", len(data))
     
        #서버에 데이터 전송
        #(str(len(stringData))).encode().ljust(16)
        s.sendall((str(len(stringData))).encode().ljust(16) + stringData)

##
        
        #if end_status:
        #    cam.release()
        #    s.close()
        #    break
     
    cam.release()



try :
    FIRST_COMM = False
    SECOND_COMM = False
    CONNECT_FLAG = False
    FIRST_CONNECT = False
    while True :
        print('START CHECKING HR04')
        gpio.output(first_trig, False)
        time.sleep(0.5)

        gpio.output(first_trig, True)
        time.sleep(0.00001)
        
        gpio.output(first_trig, False)


        while gpio.input(first_echo) == 0 :
            first_pulse_start = time.time()



        while gpio.input(first_echo) == 1 :
            first_pulse_end = time.time()


        first_pulse_duration = first_pulse_end - first_pulse_start
        first_distance = first_pulse_duration * 17000
        first_distance = round(first_distance, 2)



        gpio.output(second_trig, False)
        time.sleep(0.5)

        gpio.output(second_trig, True)
        time.sleep(0.00001)
        gpio.output(second_trig, False)

        while gpio.input(second_echo) == 0 :
            second_pulse_start = time.time()



        while gpio.input(second_echo) == 1 :
            second_pulse_end = time.time()


        second_pulse_duration = second_pulse_end - second_pulse_start
        second_distance = second_pulse_duration * 17000
        second_distance = round(second_distance, 2)
        
        print('First : ', first_distance, 'Second : ', second_distance, 'Third : ', (first_distance + second_distance)/2,'\n')
        
        if(first_distance < 40):
            if(FIRST_COMM == False):
                print(' ### First User Not Brushing ###')
            elif(FIRST_COMM == True):
                print(' ### Finished First User Brushing ###')
                FIRST_COMM = False
                CAMERA.terminate()
                pi.set_servo_pulsewidth(27, 500)
                #pi.set_servo_pulsewidth(27, 500)
                print('\n\n ### Close the Socket ###')


        else:
            print(" ### Detecting First User Brushing ###")
            if(FIRST_CONNECT == False):
                FIRST_CONNECT = True
                #pi.set_servo_pulsewidth(27, 2000)
                print("Start Socket Connect")
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                print(s)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.connect((HOST, PORT))
                print("Success Connecting")
                print(' ### First User Start Brushing ###')
                s.sendall('0'.encode())
                time.sleep(1)
                

            if(FIRST_COMM == False):
                FIRST_COMM = True
                pi.set_servo_pulsewidth(27, 2000)
                print(" ### Start Camera by First User ###")
                CAMERA = Process(target = camera)
                CAMERA.start()
############################################################################
        if(second_distance < 40):
            if(SECOND_COMM == False):
                print(' ### Second User Not Brushing ###')
            elif(SECOND_COMM == True):
                print(' ### Finished Second User Brushing ###')
                SECOND_COMM = False
                CAMERA.terminate()
                pi.set_servo_pulsewidth(27, 500)
                print('\n\n ### Close the Socket ###')


        else:
            print(" ### Detecting Second User Brushing ###")
            if(SECOND_CONNECT == False):
                SECOND_CONNECT = True
                print("Start Socket Connect")
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.connect((HOST, PORT))
                print("Success Connecting")
                print(' ### Second User Start Brushing ###')
                s.sendall('1'.encode())
                time.sleep(1)
                

            if(SECOND_COMM == False):
                SECOND_COMM = True
                pi.set_servo_pulsewidth(27, 2000)
                print(" ### Start Camera by Second User ###")
                CAMERA = Process(target = camera)
                CAMERA.start()

       
except :
    gpio.cleanup()

