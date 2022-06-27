Serving Huggingface Transformers using TorchServe

We will try to deploy pretrained model "distilbert-base-uncased-finetuned-sst-2-english" for sentiment analysis (positive and negative classes) from HuggingFace.

Installation
1.I run pure python 3.9 throw docker and install next dependencies:
apt update
apt install default-jdk 
pip install torchserve torch-model-archiver torch-workflow-archiver
pip install transformers

2.I created folder 'sentiment_deployment' and then git clone into torchserv repository.
mkdir sentiment_deployment
cd sentiment_deployment/
git clone https://github.com/pytorch/serve.git
cd serve/

Also, we need run some dependencies:
python ./ts_scripts/install_dependencies.py

We have installed all the necessary libraries.


3.Edit next file setup_config.json (serve/examples/Huggingface_Transformers)
{
 "model_name":"distilbert-base-uncased-finetuned-sst-2-english",
 "mode":"sequence_classification",
 "do_lower_case":true,
 "num_labels":"2",
 "save_mode":"pretrained",
 "max_length":"128",
 "captum_explanation":false,
 "embedding_name": "distilbert",
 "FasterTransformer":false,
 "model_parallel":false
}

4.Edit index_to_name.json
{
 "0":"Negative",
 "1":"Positive"
}

5.Lets' now download transformer model using next script:
python Download_Transformer_models.py


6.Let's create .mar file using next script (this format file understandable for torchserve)

torch-model-archiver --model-name distilBERTSeqClassification --version 1.0 --serialized-file Transformer_model/pytorch_model.bin --handler ./Transformer_handler_generalized.py --extra-files "Transformer_model/config.json,./setup_config.json,./index_to_name.json"

7.Create a directory named model_store:
mkdir model_store
mv distilBERTSeqClassification.mar model_store/ 

8.We will run torchserve as below
torchserve --start --model-store model_store --models sentiments=distilBERTSeqClassification.mar --ncs


Running inference

Before using our model we need prepare text file for predicting sentiment.(sentiment.txt) and then run next script:

curl -X POST http://127.0.0.1:8080/predictions/sentiments -T sentiment.txt

