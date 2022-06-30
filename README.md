# Serving Huggingface Transformers using TorchServe

### 1.I will deploy pretrained model "distilbert-base-uncased-finetuned-sst-2-english" for sentiment analysis (positive and negative classes) from HuggingFace.



## Installation
* 1.I run pure python 3.9 throw docker and install next dependencies:<br/>
apt update <br/>
apt install default-jdk  <br/>
pip install torchserve torch-model-archiver torch-workflow-archiver <br/>
pip install transformers <br/>

* 2.I created folder 'sentiment_deployment' and then git clone into torchserv repository.<br/>
mkdir sentiment_deployment <br/>
cd sentiment_deployment <br/>
git clone https://github.com/pytorch/serve.git <br/>
cd serve/ <br/>

Also, we need run some dependencies: <br/>
python ./ts_scripts/install_dependencies.py

We have installed all the necessary libraries.

## Edit config file
* 3.Edit next file setup_config.json (serve/examples/Huggingface_Transformers) <br/>
{ <br/>
 "model_name":"distilbert-base-uncased-finetuned-sst-2-english", <br/>
 "mode":"sequence_classification", <br/>
 "do_lower_case":true, <br/>
 "num_labels":"2", <br/>
 "save_mode":"pretrained", <br/>
 "max_length":"128",<br/>
 "captum_explanation":false, <br/>
 "embedding_name": "distilbert", <br/>
 "FasterTransformer":false, <br/>
 "model_parallel":false <br/>
}

* 4.Edit index_to_name.json
{ <br/>
 "0":"Negative", <br/>
 "1":"Positive" <br/>
} <br/>
## Building .mar file for torchserve
* 5.Lets' now download transformer model using next script: <br/>
python Download_Transformer_models.py <br/>


* 6.Let's create .mar file using next script (this format file understandable for torchserve) <br/>

torch-model-archiver --model-name distilBERTSeqClassification --version 1.0 --serialized-file Transformer_model/pytorch_model.bin --handler ./Transformer_handler_generalized.py --extra-files "Transformer_model/config.json,./setup_config.json,./index_to_name.json"

* 7.Create a directory named model_store: <br/>
mkdir model_store <br/>
mv distilBERTSeqClassification.mar model_store/  <br/>

* 8.We will run torchserve as below <br/>
torchserve --start --model-store model_store --models sentiments=distilBERTSeqClassification.mar --ncs <br/>


## Running inference

Before using our model we need prepare text file for predicting sentiment.(sentiment.txt) and then run next script: <br/>

curl -X POST http://127.0.0.1:8080/predictions/sentiments -T sentiment.txt

![Image alt](https://github.com/shaimarus/torchserve_hugging/blob/main/Negative_examples.png)
![Image alt](https://github.com/shaimarus/torchserve_hugging/blob/main/Positive_examples.png)


### 2.I will deploy pretrained resnet_18 model for image classification from HuggingFace.



* 2.I created folder 'image_classification_deployment' and then git clone into torchserv repository.<br/>
cd /home <br/>
mkdir image_classification_deployment <br/>
cd image_classification_deployment/ <br/>
git clone https://github.com/pytorch/serve.git <br/>
cd serve/ <br/>

* 3.wget https://download.pytorch.org/models/resnet18-f37072fd.pth <br/>
torch-model-archiver --model-name resnet-18 --version 1.0 --model-file ./examples/image_classifier/resnet_18/model.py --serialized-file resnet18-f37072fd.pth --handler image_classifier --extra-files ./examples/image_classifier/index_to_name.json <br/>
mkdir model_store <br/>
mv resnet-18.mar model_store/ <br/>
torchserve --start --model-store model_store --models resnet-18=resnet-18.mar <br/>
![Image alt](https://github.com/shaimarus/torchserve_hugging/blob/main/install.jpg)

* Finally <br/>
curl http://127.0.0.1:8080/predictions/resnet-18 -T ./examples/image_classifier/kitten.jpg <br/>
![Image alt](https://github.com/shaimarus/torchserve_hugging/blob/main/inference.jpg)


* ![Image alt](https://github.com/pytorch/serve/blob/master/examples/image_classifier/kitten.jpg)
