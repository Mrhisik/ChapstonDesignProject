# -*- encoding:UTF-8 -*-
from flask import Flask, render_template, request, Response, jsonify
import matplotlib.pyplot as plt
from rknnlite.api import RKNNLite
import numpy as np
from PIL import Image
import os
import serial
import serial.tools.list_ports
import time
from IPython.display import clear_output
import pandas as pd
import threading, requests, time
from threading import Event, Thread
import scipy
import sys
from scipy.signal import find_peaks
import multiprocessing
sen_num = 0
i = 0
lock = threading.Lock()
event = Event()
optime = 0
notobtime = 0
obtime = 0
bed = False
respirate = 0
heartrate = 0

def get_predict(probability):
    data = probability
    data = data.tolist()
    max_prob = max(data)

    return data.index(max_prob), max_prob

def load_model():
    # Create RKNN objects
    rknn = RKNNLite()
    # Loading RKNN model
    print('-->loading model')
    rknn.load_rknn('./sleep_moni.rknn')
    print('loading model done')
    # Initialize RKNN Running Environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
       print('Init runtime environment failed')
       exit(ret)
    print('done')
    return rknn

def predict(rknn, pic):
    im = pic   # Loading image  # Image zooming to 64x64
    outputs = rknn.inference(inputs=[im])   # Running reasoning to get reasoni>
    #pred, prob = get_predict(outputs)     # Converting Reasoning Results into >
    outputs = np.argmax(outputs)
    print(outputs)
    return outputs

def int_from_bytes(xbytes: bytes) -> int:
    temp = bytearray(xbytes)
    temp.reverse()
    temp = int.from_bytes(temp, byteorder="big", signed=False)
    return temp

def bitwise_and_bytes(a, b):
    result_int = int.from_bytes(a, byteorder="big") & int.from_bytes(b, byteorder="big")
    return result_int.to_bytes(max(len(a), len(b)), byteorder="big")

def open_serial_port(port_number, baud_rate):
    serial_port = serial.Serial(port_number, baud_rate)
    print(serial_port)
    return serial_port

def reshape(res): #11x11 크기 넘파이
    result = res[: , :64]
    results = []
    results.append(result[0:11 , 0:11])
    results = np.array(results)
    return results

def preprocess(res):
    res = res/0xffff
    result = res[: , :64]
    results = []
    results.append(result[0:11 , 0:11])
    results = np.array(results)
    return results

def timer():
    global optime
    while True:
        print(optime)
        optime = optime + 1
        time.sleep(1)
        
def sleep_timer():
    global bed
    global notobtime
    global obtime
    
    while True:
        if bed == True:
            print(obtime)
            obtime = obtime + 1
            time.sleep(1)
        else:
            print(notobtime)
            notobtime = notobtime + 1
            time.sleep(1)
            
def draw_graph(respirate_graph, h_list):
    plt.subplot(2, 1, 1)
    plt.plot(respirate_graph, color="skyblue")
    plt.title("Respiration graph")    
    plt.subplot(2,1,2)
    plt.plot(h_list, color="skyblue")
    plt.title("Heartrate graph")
    
    plt.tight_layout()
    plt.savefig("./static/images/graph.png") 
    plt.clf()

def differential(results):
    b_list = []
    dif = []
    cnt = 0
    h_list = []
    
    for i in range(results.shape[0]):
        b_list.append(results[i].mean())
    
    for i in range(len(b_list)-1):
        d = b_list[i+1]-b_list[i]
        dif.append(d)
    dif = np.array(dif)
    for i in range(dif.shape[0]): # 심박수 추출
        h_list.append(dif[i-15:i+15].mean())
    h_list = np.array(h_list)
    
    for i in range(h_list.shape[0]-1):
        if h_list[i] < 0 and h_list[i+1]>0:
            cnt+=1
    print("심박수: {0}".format(cnt*2))
    return h_list, cnt
                
def respirationrate():
    global respirate
    global heartrate
    port = "/dev/ttyUSB1"
    baud = 1497600
    exit = False
    sec = 0
    ser = serial.Serial(port, baud, timeout=None)
    bytes = []
    graph = []   
    cnt = 0
    ele = 0
    y_list = []
    graph_res = []
    graph_result = []
    graph_result1 = []
    graph_result2 = []
    
    temp = []
    
    start = time.time()
    flag = True
    while True:
        time.sleep(1/50)
        
        result = 0
        ser.write("\x41".encode())
        
        print("{0} epoch".format(cnt+1))
        for _ in range(160):

            byte = ser.read()
            i = int.from_bytes(byte, byteorder="little")

            bytes.append(i)
            result = result + i
            print(i, end=" ")
            
        if cnt == 0 and result > 0:
            cnt = cnt+ 1
            result = 0
            continue
        
        graph_res.append(result/160)
        graph_res = np.array(graph_res)
        print("\ngraph_res: {0}".format(graph_res.shape[0]))
        end = time.time()
        sec = end-start
        graph_res = graph_res.tolist()
        print(int(sec))
        if len(graph_res) > 1200:
            del graph_res[0]

        if int(sec) >= 60:
        #if int(sec) % 60 == 0 and int(sec) > 0:
            if int((int(sec) % 60) % 10) == 0:
                
                if flag == True:
                    temp = np.array(graph_res)
                    graph_res = np.array(graph_res)
                    for i in range(graph_res.shape[0]):
                        graph_result.append(graph_res[i-30:i+30].mean())
                    graph_result = np.array(graph_result)
                    for i in range(graph_result.shape[0]):
                        graph_result1.append(graph_result[i-30:i+30].mean())
                    graph_result1 = np.array(graph_result1)
                    for i in range(graph_result1.shape[0]):
                        graph_result2.append(graph_result1[i-30:i+30].mean())
                    peaks, _ = scipy.signal.find_peaks(graph_result2)
                    for j in peaks:
                        y_list.append(graph_result2[j])
                    #print("peaks: {0}".format(peaks))
                    #print("y_list: {0}".format(y_list))
                    #print(graph_result2)
                    print("peaks: {0}".format(len(peaks)))
                    
                    graph_result = graph_result.tolist()
                    graph_result1 = graph_result1.tolist()
                    
                    lock.acquire()
                    respirate = len(peaks)
                    lock.release()
                    h_list, heartrate = differential(graph_res)
                    graph_res = graph_res.tolist()
                    draw_graph(graph_result2, h_list)
                    graph_result = []
                    graph_result1 = []
                    graph_result2 = []
                    flag = False
                    continue
            else:
                flag = True
                continue
        
        
def readSensor():
    global sen_num
    global i  
    baud_rate = 115200
    mask_header = b'\xfa'
    mask_tail = b'\xfe\xfe'
    mask_serial_number = bytes(b'\x80\x00')
    port_number = '/dev/ttyUSB0'
    try:
        opened_serial_port = open_serial_port(port_number, baud_rate)     
    except:
        opened_serial_port = None
        opened_serial_port = open_serial_port(port_number, baud_rate) 
        print (opened_serial_port) 
    
    byte = opened_serial_port.read(1) 
    print(byte)

    while byte: 
        
        buffer = []
        buffers = []
        if byte == mask_header:
            temp_header = byte
            byte = opened_serial_port.read(1)

            if byte == mask_header:
                header = temp_header + byte
                
                buffer.append(int_from_bytes(header))
                byte = opened_serial_port.read(2)
                
                while byte:
                    if byte != mask_tail:
                        temp = int_from_bytes(byte)
                        buffer.append(temp)
                        
                        if len(buffer) == 2:
                            b_array = bytearray(byte)
                            b_array.reverse()
                            temp_bitwise_and = bitwise_and_bytes(b_array, mask_serial_number)
                            buffer[-1] = int_from_bytes(temp_bitwise_and)
                        
                        byte = opened_serial_port.read(2)

                    else:
                        tail = int_from_bytes(byte)
                        buffer.append(tail)
                        buffers.append(buffer)
                        break        
        #print("buffer: ", buffer)
        print("len(buffer): ", len(buffer))
        
        byte = opened_serial_port.read(1)

        buffers2 = buffers
        max_col = 32
        max_row = 64
        max_bytes = max_row * max_col
        matrix = np.zeros((len(buffers2), max_row, max_col))

        for i, buffer in enumerate(buffers2):
            pressure_sensor =buffer[2:2+max_bytes]
            matrix[i] = np.array(pressure_sensor).reshape(max_row, max_col)

        df = pd.DataFrame(matrix.reshape(-1, 32).T)
        df = df.to_numpy()
        results = df/0xffff
        print(results.shape)
        #results = results[0:11, 64*i:64*i+11]
        results = results[0:11, 0:11]
        lock.acquire()
        sen_num = results
        lock.release()
        
app = Flask(__name__)
@app.route("/", methods=['POST', 'GET'])
def mainPage():
    t1 = threading.Thread(target=readSensor)
    t2 = threading.Thread(target=timer)
    t3 = threading.Thread(target=respirationrate)
    t4 = threading.Thread(target=sleep_timer)
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    return render_template("TestPage.html", image_file="images/graph.png")

@app.route("/sen", methods=['POST', 'GET'])
def sendData():
    global sen_num
    global i
    global bed
    global obtime
    global notobtime
    global respirate
    global heartrate
    hr = heartrate
    respi = respirate
    r = sen_num

    print(" i => ", i)
    r = predict(rknn, r.astype("float32"))
    if r == 0:
        r = "nothing"
        bed = False
    elif r == 1:
        r = "leftsupine"
        bed = True
    elif r == 2:
        r = "leftprone"
        bed = True
    elif r == 3:
        r = "rightsupine"
        bed = True
    elif r == 4:
        r = "rightprone"
        bed = True
    elif r == 5:
        r = "supine"
        bed = True
    elif r == 6:
        r = "xsupine"
        bed = True
    elif r == 7:
        r = "prone"
        bed = True
    elif r == 8:
        r = "not sleeping"
        bed = False
    elif r == 9:
        r = "not sleeping"
        bed = False
        
    print(r)
    
    r = str(r)
    data = r+","+str(optime)+","+str(notobtime)+","+str(obtime)+","+str(respi)+","+str(hr)
    print(data)
    return data

if __name__ == "__main__":
    rknn=load_model()
    print("start")
    os.remove("/static/images/graph.png")
    app.run(port = 8080, debug=True, host="localhost", threaded=True)
