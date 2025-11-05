## Features

- Workflow for training & deploying an anomly detection model based on [MTAD-GAT-mlflow](https://github.com/srigas/MTAD-GAT-mlflow) for streams of sensor data, including
  - custom ML Server Docker image creation with support for CUDA & PyTorch
  - complete Docker Compose setup for accompanying data generator, data broker, queue manager, database and agent

<img width="1817" height="853" alt="architecture" src="https://github.com/user-attachments/assets/eaa0dc2b-cfc6-4c2f-af61-33b0287c1813" />


## Environment Setup

Python version 3.10 or newer needs to be installed on your machine.

The system is designed to employ a GPU for inference, therefore you need to install appropriate versions of [CUDA](https://developer.nvidia.com/cuda-toolkit-archive) and [PyTorch](https://pytorch.org/get-started/locally/).
Development was done with CUDA version 12.8 and PyTorch 1.27 under Windows, but other environments should work as well.

The mlflow tracking server is used for training and evaluating the model, as well as creating a model package for serving through ML Server.
You should use mlflow version 3.3.2 to facilitate creation of the ML Server image:

```commandline
pip install mlflow==3.3.2
```

And lastly, Docker Compose is employed to host all the services.
The recommended way to set it up is by installing [Docker Desktop](https://www.docker.com/products/docker-desktop/).
Under Linux you could alternatively install Docker / Docker Engine and Docker Compose separately as described [here](https://docs.docker.com/compose/install/).


## Workflow

### Dataset Preparation

The first step is to prepare datasets for training and evaluation.

If you have no own data you can make use of the [Controlled Anomalies Time Series](https://www.kaggle.com/datasets/patrickfleith/controlled-anomalies-time-series-dataset).
Download it from kaggle and extract the files into a folder "CATS" in the repository, like so:
```folder structure
-CATS
--data.csv
--metadata.csv
-docker
-influx
...
```

Then you can create the dataset files automatically by calling:
```commandline
ml__create_datasets.py --source CATS
```

If you want to use your own data, you need to manually create the following files and folders:
```folder structure
-datasets
--<your_model_name>
---eval.csv
---eval_labels.csv
---train.csv
---train_labels.csv
```
Where the eval/train.csv contain your comma-separated sensor values, without headers.
And the eval/train_labels.csv contain a matching amount of lines, with a single 0 or 1 each (to indicate an anomaly in the sensor values).


### Model Training and Evaluation

To train your model call:
```commandline
ml__train_model.py
```

You can pass a parameter "dataset" matching the name of the folder under "datasets" created in the first step.
The rest of the parameters are related to the MTAD-GAT model and you'll need to adjust them according to your data until you're satisfied with the model metrics.

For each training iteration mlflow will create a new run for your model.

The next step is to evaluate your model:
```commandline
ml__evaluate_model.py
```

Don't skip this step, as it will find an appropriate anomaly threshold for your data, and mlflow will update your model artifacts.


### Docker Image Creation

Make sure to start docker / the docker daemon, before attempting the next steps.


The packaged mlflow model is going to be served by an ML Server instance.

To make use of the GPU for inference, you first need to create a general ML Server docker image matching your development environment:
```commandline
docker__build_mlflow_base_image.py
```

If you're using different versions for anything installed in the "Environment Setup", you'll need to adjust the parameters for this call.
Be aware that you also have to adjust the "ubuntu_version" if you use a different version of CUDA (to match exisiting nvidia docker images), and the "fastapi_version" if you use a different mlflow version.

Note: Unless you change your environment setup, you only have to create this "base" image once.


For the mlflow model a separate image will be created from this base image with:
```commandline
docker__build_detector_image.py
```

At this point you have created a docker image, that is ready to serve your model on the GPU, via the ML Server APIs as described [here](https://docs.seldon.ai/mlserver/examples/mlflow#send-test-inference-request).
So, if you have an existing application, you could go integrate it as a service, and stop the workflow at this step.


If you want a fully integrated model / test environment, you'll need to create two more docker images.

One for the data generator, which turns the tabular test data into a stream:
```commandline
docker__build_generator_image.py
```

And one for the queue manager, which handles queuing the sensor data and calling the ML Server with the windows of data, to return an anomaly score:
```commandline
docker__build_manager_image.py
```

Note that these two are not dependent on the machine learning model.
That is, you only need to recreate the generator image, if you change the number of features, or want to use different test data.
And the manager image only ever needs to be recreated, if you change its source code (under "mqtt").


### System Configuration

Alongside the custom docker images created above, the system also includes
- a mosquitto mqtt broker to receive the sensor readings and anomaly detections
- an influxdb database (V2) to store sensor readings, anomaly detections, and ML Server metrics
- a telegraf agent for pulling mqtt messages as well as ML Server metrics and funneling them to the influxdb database


The last step is to configure this compositon of services to form an application of your liking.
This is done through the ".env", "docker-compose.yml", and "set_authentication.py" in the "docker" subfolder.

The ".env" file is used to set recurring values like your company/repository name, and settings that will likely be different between development and production environment.

In the "docker-compose.yml" you can configure a wealth of settings for the broker, database, agent, as well as the three custom images.
The main thing you should do here is to set the "THRESHOLD" of the manager to the best result of your model evaluation.

Otherwise, if you have called all previous python files with their default parameters, you really only need to create docker secrets with:
```commandline
set_authentication.py --mosquitto_sensor_pass <password> --mosquitto_manager_pass <password> --mosquitto_telegraf_pass <password> --influxdb_pass <password>
```

In case this command failes to create a "mosquitto_users" secret, try manually pulling the eclipse-mosquitto docker image first:
```commandline
docker pull eclipse-mosquitto:2.0.20
```
A temporary container is created from this image to initialize the users file.


### Startup

Finally you can start the system from within the "docker" subfolder:
```commandline
docker compose up
```

This will create the containers for all the services, and start them up in the appropriate order.
In Docker Desktop, the result should look like this:

<img width="791" height="424" alt="containers" src="https://github.com/user-attachments/assets/3cbdbaca-8717-43a5-b945-3c1d42140ad4" />

Now you should be able to access the influxdb frontend via: http://localhost:8086.
Use "admin" and the password you have set with the "set_authentication.py" to login.
The incoming sensor data as well as anomaly scores, can be found in the "Data Explorer".


### Creating an influxdb dashboard 

To create a proper dashboard for the incoming data, you can use the templates in the "influx" folder of the repository.
Simply drag and drop the template files for the two variables, and the dashboard itself on the respective "import" window.
This should create a dashboard called "Sensors", looking similar to this:

<img width="1196" height="500" alt="dashboard" src="https://github.com/user-attachments/assets/c50ac7b7-8cbc-44b3-b679-d177f2c20c57" />


You might need to manually edit the graphs though, and set the correct values for the Y Axis Domain, as these are often not imported properly:

<img width="461" height="484" alt="custom_domain" src="https://github.com/user-attachments/assets/73bc645f-3005-49e9-b11f-58bbb2610019" />



### Retraining

At a later point, you can use the data accumulated in the influxdb (in your production environment) to adapt your model.
For this, open the "Data Explorer", filter the desired machine / sensors / timeframe, press "Export Data" and choose the "csv" format.
Since you're probably retraining because the model predictions aren't matching your encountered anomalies anymore, you'll need to create a separate csv containing the latter.
This only needs to contain two columns "start_time" and "end_time" (similar to the first two columns of the CATS metadata.csv).

You can then create new datasets for training and evaluation via:
```commandline
ml__create_datasets.py --source influx --data_csv <path_to_the_influxdb_export> --anomalies_csv <path_to_the_anomalies_csv>
```

With these datasets you can follow the steps for training and evaluation again, recreate the generator and detector images, update the threshold in the .yml, and restart the docker composition to test your new model.


## Disclaimer

**No** part of the source code in this repository or this documentation was created by or with the help of artificial intelligence.
