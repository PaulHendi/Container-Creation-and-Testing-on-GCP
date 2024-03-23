LOCATION=us-east4
PROJECT=hf-notebooks
REPO_NAME=gpu-transformers
IMAGE_NAME=gpu_transformers

docker build -t $IMAGE_NAME:v1 --platform linux/amd64 . 

gcloud auth configure-docker $LOCATION-docker.pkg.dev

docker tag $IMAGE_NAME:v1 $LOCATION-docker.pkg.dev/$PROJECT/$REPO_NAME/$IMAGE_NAME:v1

docker push $LOCATION-docker.pkg.dev/$PROJECT/$REPO_NAME/$IMAGE_NAME:v1
