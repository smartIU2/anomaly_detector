import io
import os
import argparse
import random
import string
import docker
import tarfile


MOSQUITTO_ACL_TEMPLATE = """user {sensor_user}
topic write {topic_org}/sensor/{topic_sensor}

user {manager_user}
topic read {topic_org}/sensor/{topic_sensor}
topic write {topic_org}/anomaly/{topic_sensor}

user {telegraf_user}
topic read {topic_org}/sensor/#
topic read {topic_org}/anomaly/#
"""

def write_ACL_file(sensor_user, manager_user, telegraf_user, topic_org, topic_sensor):
    with open("./secrets/mosquitto_acl", "w") as f:
        f.write(
            MOSQUITTO_ACL_TEMPLATE.format(
                sensor_user=sensor_user,
                manager_user=manager_user,
                telegraf_user=telegraf_user,
                topic_org=topic_org,
                topic_sensor=topic_sensor
            )
        )

def write_secret(secret, content):
    with open(os.path.join("./secrets", secret), "w") as f:
        f.write(content)

def generate_mosquitto_users_file(users):
    
    # init docker client
    client = docker.from_env()
    
    # create temp mosquitto container   
    container = client.containers.run("eclipse-mosquitto:2.0.20"
                                     ,auto_remove=True
                                     ,detach=True)
    
    # create users file inside the container
    users_file = "mosquitto_users"
    container.exec_run(["touch", users_file])
    
    # add users
    for user in users:
        container.exec_run(["mosquitto_passwd", "-b",
                            users_file, user[0], user[1]])
    
    # copy file to host
    f, _ = container.get_archive(users_file)
    b = io.BytesIO(next(f))
    with tarfile.open(mode='r', fileobj=b) as tar:
        tar.extractall(path="./secrets/")
      
    # shutdown temp container
    container.stop()


if __name__ == "__main__":

    # get arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--mosquitto_sensor_user", type=str, default="sensor_blade"
                      , help="user to write sensor data to mosquitto")
    parser.add_argument("--mosquitto_sensor_pass", type=str, required=True)
    parser.add_argument("--mosquitto_manager_user", type=str, default="manager_blade"
                      , help="user to write anomaly data to mosquitto")
    parser.add_argument("--mosquitto_manager_pass", type=str, required=True)
    parser.add_argument("--mosquitto_telegraf_user", type=str, default="telegraf"
                      , help="user to read sensor data from mosquitto")
    parser.add_argument("--mosquitto_telegraf_pass", type=str, required=True)
    parser.add_argument("--mosquitto_topic_org", type=str, default="windmill")
    parser.add_argument("--mosquitto_topic_sensor", type=str, default="blade")
    parser.add_argument("--influxdb_user", type=str, default="admin"
                      , help="influxdb admin user")
    parser.add_argument("--influxdb_pass", type=str, required=True)
    
    args = parser.parse_args()
    

    #generate influxdb access token
    token = f"{''.join(random.choices(string.ascii_letters + string.digits, k=13))}0=="
    
    #write secrets
    os.makedirs("./secrets", exist_ok=True)
    
    write_secret("mosquitto_sensor_user", args.mosquitto_sensor_user)
    write_secret("mosquitto_sensor_pass", args.mosquitto_sensor_pass)
    write_secret("mosquitto_manager_user", args.mosquitto_manager_user)
    write_secret("mosquitto_manager_pass", args.mosquitto_manager_pass)
    write_secret("mosquitto_telegraf_user", args.mosquitto_telegraf_user)
    write_secret("mosquitto_telegraf_pass", args.mosquitto_telegraf_pass)
    write_secret("influxdb_user", args.influxdb_user)
    write_secret("influxdb_pass", args.influxdb_pass)
    write_secret("influxdb_token", token)
    
    #write mosquitto ACL file
    write_ACL_file(args.mosquitto_sensor_user, args.mosquitto_manager_user, args.mosquitto_telegraf_user, args.mosquitto_topic_org, args.mosquitto_topic_sensor)

    #generate mosquitto pwd file
    generate_mosquitto_users_file([(args.mosquitto_sensor_user,args.mosquitto_sensor_pass),
                                   (args.mosquitto_manager_user,args.mosquitto_manager_pass),
                                   (args.mosquitto_telegraf_user,args.mosquitto_telegraf_pass)])
                   
                   
    print("Succesfully created docker secrets files under './secrets'")
