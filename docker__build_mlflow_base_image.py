import os
import argparse

from mlflow.models import docker_utils


DOCKERFILE_TEMPLATE = """FROM docker.io/nvidia/cuda:{CUDA_VERSION}.0-base-ubuntu{UBUNTU_VERSION}

LABEL description="Base for building images that can serve mlflow models with CUDA."

# install commons
RUN apt-get -y update \
 && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC \
 && apt-get install -y --no-install-recommends software-properties-common wget curl nginx ca-certificates bzip2 build-essential cmake git-core

# install python & pip (using deadsnakes for archived versions)
RUN add-apt-repository -y ppa:deadsnakes/ppa \
 && apt-get update
RUN apt-get purge -y --autoremove software-properties-common 
RUN apt-get install -y python{PYTHON_VERSION} python{PYTHON_VERSION}-distutils \
 && ln -s -f $(which python{PYTHON_VERSION}) /usr/bin/python
RUN alias python=/usr/bin/python
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python{PYTHON_VERSION}
RUN mkdir -p /pip_cache
ENV PIP_CACHE_DIR=/pip_cache

# install mlflow
WORKDIR /opt/mlflow
RUN pip install fastapi=={FASTAPI_VERSION} mlflow[mlserver]=={MLFLOW_VERSION} --ignore-installed --break-system-packages

ENV MLFLOW_DISABLE_ENV_CREATION=true
ENV ENABLE_MLSERVER=true

# granting read/write access and conditional execution authority to all child directories
# and files to allow for deployment to AWS Sagemaker Serverless Endpoints
# (see https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html)
RUN chmod o+rwX /opt/mlflow/

# install torch
RUN pip install torch=={TORCH_VERSION} --ignore-installed --break-system-packages --index-url https://download.pytorch.org/whl/cu{CUDA_VERSION_WHEEL}
"""


def generate_dockerfile(
    output_dir: str,
    CUDA_VERSION: str,
    UBUNTU_VERSION: str,
    PYTHON_VERSION: str,
    TORCH_VERSION: str,
    FASTAPI_VERSION: str,
    MLFLOW_VERSION: str
):

    with open(os.path.join(output_dir, "Dockerfile"), "w") as f:
        f.write(
            DOCKERFILE_TEMPLATE.format(
                CUDA_VERSION=CUDA_VERSION,
                CUDA_VERSION_WHEEL=CUDA_VERSION.replace('.',''),
                UBUNTU_VERSION=UBUNTU_VERSION,
                PYTHON_VERSION=PYTHON_VERSION,
                TORCH_VERSION=TORCH_VERSION,
                FASTAPI_VERSION=FASTAPI_VERSION,
                MLFLOW_VERSION=MLFLOW_VERSION
            )
        )


if __name__ == "__main__":

    # get arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=str, default="windmill")
    parser.add_argument("--mlflow_version", type=str, default="3.3.2")
    parser.add_argument("--cuda_version", type=str, default="12.8")
    parser.add_argument("--ubuntu_version", type=str, default="24.04")
    parser.add_argument("--python_version", type=str, default="3.10")
    parser.add_argument("--torch_version", type=str, default="2.7.0")
    parser.add_argument("--fastapi_version", type=str, default="0.115.14")
    
    args = parser.parse_args()


    # create output dir
    output_dir = os.path.join("docker", "images", "mlflow-cuda-base")
    os.makedirs(output_dir, exist_ok=True)


    # create docker image
    generate_dockerfile(output_dir,
        args.cuda_version, args.ubuntu_version, args.python_version, args.torch_version, args.fastapi_version, args.mlflow_version)
    
    docker_utils.build_image_from_context(output_dir, f"{args.repository.lower()}/mlflow-cuda-base:{args.mlflow_version}-{args.cuda_version}")