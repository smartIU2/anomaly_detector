import argparse
import os

from mlflow.models.flavor_backend_registry import get_flavor_backend
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.models import docker_utils


DOCKERFILE_TEMPLATE = """FROM {base_image}

LABEL maintainer="{organization}"
LABEL description="Serve '{model_name}' model using mlflow / mlserver, with CUDA support."

{install_model_and_deps}

# clean up apt & pip cache to reduce image size
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*
RUN rm -rf /pip_cache/*

ENTRYPOINT ["python", "-c", "from mlflow.models import container as C; C._serve('local')"]
"""

def generate_dockerfile(
    output_dir: str,
    organization: str,
    base_image: str,
    model_name: str,
    install_steps: str
):

    with open(os.path.join(output_dir, "Dockerfile"), "w") as f:
        f.write(
            DOCKERFILE_TEMPLATE.format(
                organization=organization,
                base_image=base_image,
                model_name=model_name,
                install_model_and_deps=install_steps
            )
        )

def get_install_steps(
    output_dir: str,
    model_name: str
):

    model_dir = os.path.join(output_dir, "model_dir")
    os.makedirs(model_dir, exist_ok=True)

    model_uri = f"models:/{model_name}_model/latest"
    mlflow_backend = get_flavor_backend(model_uri, docker_build=True, env_manager="local")

    model_path = _download_artifact_from_uri(model_uri, output_path=model_dir)

    copy_src = os.path.relpath(model_path, start=output_dir)
    
    return mlflow_backend._model_installation_steps(
        copy_src,
        model_path,
        "local",
        False,
        False
    )
    

if __name__ == "__main__":

    # get arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=str, default="windmill")
    parser.add_argument("--base_image", type=str, default="windmill/mlflow-cuda-base:3.3.2-12.8")
    parser.add_argument("--model_name", type=str, default="blade")
    
    args = parser.parse_args()

    repository = args.repository.lower()
    model_name = args.model_name.lower()
    image_name = f"{model_name}-detector"


    # create output dir
    output_dir = os.path.join("docker", "images", image_name)
    os.makedirs(output_dir, exist_ok=True)


    # generate mlflow / mlserver setup commands
    install_steps = get_install_steps(output_dir, model_name)


    # create docker image
    generate_dockerfile(output_dir, repository, args.base_image, model_name, install_steps)
    
    docker_utils.build_image_from_context(output_dir, f"{repository}/{image_name}")