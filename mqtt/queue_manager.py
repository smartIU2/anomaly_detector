import os
import json
import requests
import paho.mqtt.client as mqtt

from collections import deque


INVALID_DATA_MESSAGE = "Received sensor data '{reading}' does not match influx line procotol format."


class queue_manager:

    def __init__(self, mqtt_server, mqtt_usr, mqtt_pass, sub_topic, sensors, detector_url, window_size, pub_topic, qos, threshold):
        
        # config
        self.mqtt_server = mqtt_server
        self.sub_topic = sub_topic
        self.sensors = sensors.split(',')
        self.detector_url = detector_url
        self.window_size = window_size
        self.pub_topic = pub_topic
        self.qos = qos
        self.threshold = threshold
        
        # dict of message queues
        self.message_queues = {}
        
        # mqtt client
        self.mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.mqttc.on_connect = self.on_connect
        self.mqttc.on_message = self.on_message
        self.mqttc.on_publish = self.on_publish
        
        # authentication
        self.mqttc.username_pw_set(mqtt_usr, mqtt_pass)
        
    
    # connect to mqtt broker
    def connect(self):
        self.mqttc.connect(self.mqtt_server, 1883, 60)
        self.mqttc.loop_forever()
        
    # subscribe on connect
    def on_connect(self, client, userdata, flags, reason_code, properties):
        print(f"Connected with result code {reason_code}")
        client.subscribe(self.sub_topic)


    # handle incoming messages
    def on_message(self, client, userdata, msg):

        #parse sensor reading from influxdb line protocol
        #(measurement,tag=tag field=field timestamp)

        reading = str(msg.payload)
        msg_parts = reading.strip("'").split(' ')

        assert len(msg_parts) == 3, INVALID_DATA_MESSAGE.format(reading=reading)  
        machine = msg_parts[0].split('=')
        
        assert len(machine) == 2, INVALID_DATA_MESSAGE.format(reading=reading)
        machine = machine[1]
        
        assert msg_parts[2].isdecimal(), INVALID_DATA_MESSAGE.format(reading=reading) 
        # timestamp of sensor data reading
        # timestamp for the anomaly detection equals last data received for a set of readings 
        timestamp = int(msg_parts[2])
        
        sensor = msg_parts[1].split('=')
        
        assert len(sensor) == 2 and sensor[1].lstrip('-').replace('.', '', 1).isdecimal(), INVALID_DATA_MESSAGE.format(reading=reading)
        value = float(sensor[1])
        sensor = sensor[0]


        # construct a separate queue for each machine
        if machine not in self.message_queues:            
            self.message_queues[machine] = deque(maxlen=self.window_size + 1)
        
        queue = self.message_queues[machine]
        
        
        # set sensor values
        if (len(queue) == 0) or (not None in queue[-1]):
            queue.append([None] * len(self.sensors))
        
        reading = queue[-1]
        
        reading[self.sensors.index(sensor)] = value
        
        
        # detect anomalies, after queuing enough messages
        if (not None in reading):
        
            if len(queue) == self.window_size + 1:
                print(f"Detecting anomalies for machine '{machine}'.")
                
                anomaly = self.detect_anomalies(list(queue))
                
                #publish anomaly detection result to mosquitto
                self.mqttc.publish(self.pub_topic, f'sensor,machine={machine} anomaly={anomaly} {timestamp}', qos=self.qos)
                
            else:
                print(f"Queuing message for machine '{machine}' until window_size is reached.")


    # pass readings to ML model for anomaly detection
    def detect_anomalies(self, readings):
        
        if self.threshold is not None:
            # pass custom threshold
            inputs = json.dumps({"inputs": [readings],
                                 "params": {
                                    "mode": "anomaly",
                                    "threshold": float(self.threshold)
                                 }})
        else:
            # use default (epsilon) threshold
            inputs = json.dumps({"inputs": [readings]})
        
        
        response = requests.post(
            url=self.detector_url,
            data=inputs,
            headers={"Content-Type": "application/json"},
        )

        return response.json()["predictions"][0]


    # simple logging
    def on_publish(self, client, userdata, mid, reason_code, properties):
        print(f"Anomaly detection {mid} published with result code {reason_code}")



if __name__ == "__main__":

    # get user & pass from docker secrets
    with open(os.getenv('MQTT_USER_FILE'), 'r') as file:
        usr = file.read() 

    with open(os.getenv('MQTT_PASS_FILE'), 'r') as file:
        pwd = file.read()  

    # start manager
    manager = queue_manager(os.getenv('MQTT_SERVER'), usr, pwd
                           ,os.getenv('SUB_TOPIC')
                           ,os.getenv('SENSORS')
                           ,os.getenv('DETECTOR')
                           ,int(os.getenv('WINDOW_SIZE'))
                           ,os.getenv('PUB_TOPIC')
                           ,int(os.getenv('QOS'))
                           ,os.getenv('THRESHOLD'))
    
    manager.connect()