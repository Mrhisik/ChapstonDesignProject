# -*- encoding:UTF-8 -*-
from flask import Flask, render_template, request, Response
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

sen_num = 0
i = 0
lock = threading.Lock()
event = Event()

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
    buffers = []
    byte = opened_serial_port.read(1) 
    print(byte)
    
    while byte: 
        buffer = []
        
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

        seconds = time.time()

        print("Seconds since epoch =", seconds)	
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
    t1.start()
    return render_template("TestPage.html")

@app.route("/sen", methods=['POST', 'GET'])
def sendData():
    global sen_num
    global i

    r = sen_num

    print(" i => ", i)
    r = predict(rknn, r.astype("float32"))
    if r == 0:
        r = "nothing"
    elif r == 1:
        r = "leftsupine"
    elif r == 2:
        r = "leftprone"
    elif r == 3:
        r = "rightsupine"
    elif r == 4:
        r = "rightprone"
    elif r == 5:
        r = "supine"
    elif r == 6:
        r = "Playing"
    print(r)
    r = str(r)
    return r
 
if __name__ == "__main__":
    rknn=load_model()
    print("start")

    app.run(port = 8080, debug=True, host="localhost", threaded=True)

    
