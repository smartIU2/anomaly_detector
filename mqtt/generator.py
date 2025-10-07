import os
import pandas as pd
import random
import time
import paho.mqtt.client as mqtt


class sensor_generator:

    def __init__(self, mqtt_server, mqtt_usr, mqtt_pass, pub_topic, interval, qos, machine, mock_data):
        
        # config
        self.mqtt_server = mqtt_server
        self.pub_topic = pub_topic
        self.interval = interval
        self.qos = qos
        self.machine = machine
        self.mock_data = mock_data
        
        # mqtt client
        self.mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.mqttc.on_connect = self.on_connect
        self.mqttc.on_publish = self.on_publish
        
        # authentication
        self.mqttc.username_pw_set(mqtt_usr, mqtt_pass)
        
        
    # publish mock sensor data every {interval} seconds
    def stream(self):
 
        self.mqttc.connect(self.mqtt_server, 1883, 60)
        self.mqttc.loop_start()
 
        for _, reading in self.mock_data.iterrows():

            timestamp = time.time_ns()
            for sensor in self.mock_data.columns:
            
                #format sensor reading to influxdb line protocol
                #(measurement,tag=tag field=field timestamp)
                self.mqttc.publish(self.pub_topic, f'sensor,machine={self.machine} {sensor}={reading[sensor]} {timestamp}', qos=self.qos)

            time.sleep(self.interval)


        # close connection upon reaching end of data
        self.mqttc.loop_stop()
        self.mqttc.disconnect()

    # simple logging
    def on_connect(self, client, userdata, flags, reason_code, properties):
        print(f"Connected with result code {reason_code}")
            
    def on_publish(self, client, userdata, mid, reason_code, properties):
        print(f"Sensor reading {mid} published with result code {reason_code}")
      
      

if __name__ == "__main__":

    # get user & pass from docker secrets
    with open(os.getenv('MQTT_USER_FILE'), 'r') as file:
        usr = file.read() 

    with open(os.getenv('MQTT_PASS_FILE'), 'r') as file:
        pwd = file.read()  

    # get id from container, so the same image can be run multiple times without changing any configs
    machine = os.getenv('HOSTNAME')

    # read mock sensor data
    # from random starting row, so not all sensors for a model publish the same data
    mock_data = pd.read_csv('data.csv', index_col=False, skiprows=range(1,random.randint(2,10000)))


    # start stream
    generator = sensor_generator(os.getenv('MQTT_SERVER'), usr, pwd
                                ,os.getenv('PUB_TOPIC')
                                ,float(os.getenv('INTERVAL'))
                                ,int(os.getenv('QOS'))
                                ,machine
                                ,mock_data)
    
    generator.stream()