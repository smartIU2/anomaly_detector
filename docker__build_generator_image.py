import argparse
import os
import shutil

from mlflow.models import docker_utils


DOCKERFILE_TEMPLATE = """FROM python:{PYTHON_VERSION}-alpine

LABEL maintainer="{organization}"
LABEL description="Publish mock messages for '{model_name}' model to mqtt server, using influx line protocol."

RUN pip install paho-mqtt pandas

WORKDIR /app
COPY /app/data.csv .
COPY /app/generator.py .

ENTRYPOINT ["python", "generator.py"]
"""

def generate_dockerfile(
    output_dir: str,
    PYTHON_VERSION: str,
    organization: str,
    model_name: str
):

    with open(os.path.join(output_dir, "Dockerfile"), "w") as f:
        f.write(
            DOCKERFILE_TEMPLATE.format(
                PYTHON_VERSION=PYTHON_VERSION,
                organization=organization,
                model_name=model_name
            )
        )

def generate_data(
    output_dir: str,
    data_path: str,
    headers: str
):

    with open(data_path, 'r') as file:
        content = file.read()

    with open(os.path.join(output_dir, "app", "data.csv"), "w") as file:
        file.write(headers + '\n' + content)


if __name__ == "__main__":

    # get arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--python_version", type=str, default="3.10")
    parser.add_argument("--repository", type=str, default="windmill")
    parser.add_argument("--model_name", type=str, default="blade")
    parser.add_argument("--mock_data", type=str, default="./datasets/blade/eval.csv")
    parser.add_argument("--sensors", type=str, default="temperature,volume,humidity")
   
    args = parser.parse_args()

    repository = args.repository.lower()
    model_name = args.model_name.lower()
    image_name = f"{model_name}-generator"


    # create output dir
    output_dir = os.path.join("docker", "images", image_name)
    os.makedirs(os.path.join(output_dir, "app"), exist_ok=True)


    # copy code
    shutil.copy("./mqtt/generator.py", os.path.join(output_dir, "app", "generator.py"))

    
    # create mock data csv
    generate_data(output_dir, args.mock_data, args.sensors)
    
    
    # create docker image
    generate_dockerfile(output_dir, args.python_version, repository, model_name)
    
    docker_utils.build_image_from_context(output_dir, f"{repository}/{image_name}")