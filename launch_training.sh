# Change those two variables to the appropriate values
CONTAINER_NAME=us-east4-docker.pkg.dev/hf-notebooks/gpu-transformers/gpu_transformers:v1
VOLUME=~/hf-docker-app-gpu-transformer/volume

# Configure Docker to use the gcloud command-line tool as a credential helper 
gcloud auth configure-docker us-east4-docker.pkg.dev

#Create the app directory
echo "Creating the app_mount directory ..."
mkdir -p ${VOLUME}


# Choose the training script
if [ $1 = "gpt2" ]; then
    echo "Training GPT2 model on a language modelling task..."
    SCRIPT=/volume/test_training_distillgpt2.py
else 
    # Default to bert base
    echo "Training BERT base model on a classification task..."
    SCRIPT=/app/test_training_bert_base.py
fi

#Launch the container and run the training script
echo "Launching the container and running the training script ..."
docker run --gpus all -v ${VOLUME}:/volume ${CONTAINER_NAME} python3 $SCRIPT
