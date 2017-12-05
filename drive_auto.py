import urllib
import cv2
import numpy as np
import serial
import math

# enter url'
url = ''

class NeuralNetwork(object):

    def __init__(self):
        self.model = cv2.ANN_MLP()

    def create(self):
        layer_size = np.int32([38400, 32, 4])
        self.model.create(layer_size)
        self.model.load('computer/mlp_xml/mlp4.xml')

    def predict(self, samples):
        # print(samples)
        ret, resp = self.model.predict(samples)
        return resp.argmax(-1)

class RCControl(object):

    def __init__(self):
        print('move')
        self.serial_port = serial.Serial('COM6', 115200, timeout=1)

    def steer(self, prediction):
        if prediction == 2:
            self.serial_port.write(chr(1))
            print("Forward")
        elif prediction == 0:
            self.serial_port.write(chr(7))
            # self.serial_port.write(chr(1))
            print("Left")
        elif prediction == 1:
            self.serial_port.write(chr(6))
            # self.serial_port.write(chr(1))
            print("Right")
        else:
            # pass
            self.stop()

    def stop(self):
        print('Stop')
        self.serial_port.write(chr(0))

class fetchStream():

    model = NeuralNetwork()
    model.create()
    rc_car = RCControl()

    def __init__(self):

        self.stream = urllib.urlopen(url)
        self.fetchFrame()

    def fetchFrame(self):

        try:
            stream_bytes = ''

            while True:

                stream_bytes += self.stream.read(1024)
                first = stream_bytes.find('\xff\xd8')
                last = stream_bytes.find('\xff\xd9')

                if last != -1 and first != -1:
                    # print('Got it')
                    jpg = stream_bytes[first:last+2]
                    stream_bytes = stream_bytes[last+2:]
                    gray = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.CV_LOAD_IMAGE_GRAYSCALE)
                    image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.CV_LOAD_IMAGE_COLOR)
                    # print(image.shape)
                    cv2.imshow('Stream', image)
                    cv2.imshow('Input', gray)

                    half_gray = gray[120:240, :]
                    cv2.imshow('Input', half_gray)
                    # print(image.shape())

                    image_array = half_gray.reshape(1, 38400).astype(np.float32)

                    prediction = self.model.predict(image_array)

                    self.rc_car.steer(prediction)

                    if cv2.waitKey(1) == 27:
                        exit(0)


        except:
            pass
            print('Something wrong')

if __name__ == '__main__':
    fetchStream()


cv2.destroyAllWindows()

