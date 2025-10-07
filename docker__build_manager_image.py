import os
import argparse
import shutil

from mlflow.models import docker_utils


DOCKERFILE_TEMPLATE = """FROM python:{PYTHON_VERSION}-alpine

LABEL maintainer="{organization}"
LABEL description="Manage mqtt message queues for mtad_gat ML model."

RUN pip install paho-mqtt requests

WORKDIR /app
COPY /app/queue_manager.py .

ENTRYPOINT ["python", "queue_manager.py"]
"""

def generate_dockerfile(
    output_dir: str,
    PYTHON_VERSION: str,
    organization: str
):

    with open(os.path.join(output_dir, "Dockerfile"), "w") as f:
        f.write(
            DOCKERFILE_TEMPLATE.format(
                PYTHON_VERSION=PYTHON_VERSION,
                organization=organization
            )
        )


if __name__ == "__main__":

    # get arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--python_version", type=str, default="3.10")
    parser.add_argument("--repository", type=str, default="windmill")
    
    args = parser.parse_args()

    repository = args.repository.lower()
    image_name = "queue-manager"


    # create output dir
    output_dir = os.path.join("docker", "images", image_name)
    os.makedirs(os.path.join(output_dir, "app"), exist_ok=True)


    # copy code
    shutil.copy("./mqtt/queue_manager.py", os.path.join(output_dir, "app", "queue_manager.py"))


    # create docker image
    generate_dockerfile(output_dir, args.python_version, repository)
    
    docker_utils.build_image_from_context(output_dir, f"{repository}/{image_name}")