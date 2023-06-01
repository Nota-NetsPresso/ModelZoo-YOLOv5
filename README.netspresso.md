<div align="center">
  <p>
    <a align="center" href="README.netspresso.md" target="_blank">
      <img width="100%" src="./imgs/banner2.png"></a>
  </p>

</div>

# <div align="center">NetsPresso Tutorial for YOLOv5 Compressrion</div>
## Order of the tutorial
[1. Install](#1-install) </br>
[2. Training](#2-training) </br>
[3. Convert YOLOv5 to _torchfx.pt](#3-convert-yolov5-to-_torchfxpt) </br>
[4. Model Compression with NetsPresso Python Package](#4-model-compression-with-netspresso-python-package)</br>
[5. Fine-tuning the compressed Model](#5-fine-tuning-the-compressed-model)</br>
</br>

## 1. Install
Clone repo and install [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) in a
[**Python>=3.7.0**](https://www.python.org/) environment, including
[**PyTorch>=1.11**](https://pytorch.org/get-started/locally/).

```bash
git clone https://github.com/Nota-NetsPresso/yolov5_nota.git  # clone
cd yolov5_nota
pip install -r requirements.txt  # install
```
</br>

## 2. Training
If you want to start from scratch, create a '.pt' file via 'train.py'.
```bash
python train.py --data coco.yaml --epochs 300 --weights '' --cfg yolov5n.yaml  --batch-size 128
                                                                 yolov5s                    64
                                                                 yolov5m                    40
                                                                 yolov5l                    24
                                                                 yolov5x                    16
```
</br>

## 3. Convert YOLOv5 to _torchfx.pt
```bash
python export.py --weights yolov5s.pt --include netspresso
```
Executing this code will create 'model_torchfx.pt' and 'model_head.pt'.<br/><br/>

## 4. Model Compression with NetsPresso Python Package<br/>
Upload & compress your 'model_torchfx.pt' by using NetsPresso Python Package
### 4_1. Install NetsPresso Python Package
```bash
pip install netspresso
```
### 4_2. Upload & Compress
First, import the packages and set a NetsPresso username and password.
```python
from netspresso.compressor import ModelCompressor, Task, Framework, CompressionMethod, RecommendationMethod


EMAIL = "YOUR_EMAIL"
PASSWORD = "YOUR_PASSWORD"
compressor = ModelCompressor(email=EMAIL, password=PASSWORD)
```
Second, upload 'model_torchfx.pt', which is the model converted to torchfx in step 3, with the following code.
```python
# Upload Model
UPLOAD_MODEL_NAME = "yolov5_model"
TASK = Task.OBJECT_DETECTION
FRAMEWORK = Framework.PYTORCH
UPLOAD_MODEL_PATH = "./model_torchfx.pt"
INPUT_LAYERS = [{"batch": 1, "channel": 3, "dimension": [640, 640]}]
model = compressor.upload_model(
    model_name=UPLOAD_MODEL_NAME,
    task=TASK,
    framework=FRAMEWORK,
    file_path=UPLOAD_MODEL_PATH,
    input_layers=INPUT_LAYERS,
)
```
Finally, you can compress the uploaded model with the desired options through the following code.
```python
# Recommendation Compression
COMPRESSED_MODEL_NAME = "test_l2norm"
COMPRESSION_METHOD = CompressionMethod.PR_L2
RECOMMENDATION_METHOD = RecommendationMethod.SLAMP
RECOMMENDATION_RATIO = 0.6
OUTPUT_PATH = "./compressed_yolov5.pt"
compressed_model = compressor.recommendation_compression(
    model_id=model.model_id,
    model_name=COMPRESSED_MODEL_NAME,
    compression_method=COMPRESSION_METHOD,
    recommendation_method=RECOMMENDATION_METHOD,
    recommendation_ratio=RECOMMENDATION_RATIO,
    output_path=OUTPUT_PATH,
)
```

<details>
<summary>Click to check 'Full Upload&Compress Code'</summary>

```bash
pip install netspresso
```

```python
from netspresso.compressor import ModelCompressor, Task, Framework, CompressionMethod, RecommendationMethod


EMAIL = "YOUR_EMAIL"
PASSWORD = "YOUR_PASSWORD"
compressor = ModelCompressor(email=EMAIL, password=PASSWORD)

# Upload Model
UPLOAD_MODEL_NAME = "yolov5_model"
TASK = Task.OBJECT_DETECTION
FRAMEWORK = Framework.PYTORCH
UPLOAD_MODEL_PATH = "./model_torchfx.pt"
INPUT_LAYERS = [{"batch": 1, "channel": 3, "dimension": [640, 640]}]
model = compressor.upload_model(
    model_name=UPLOAD_MODEL_NAME,
    task=TASK,
    framework=FRAMEWORK,
    file_path=UPLOAD_MODEL_PATH,
    input_layers=INPUT_LAYERS,
)

# Recommendation Compression
COMPRESSED_MODEL_NAME = "test_l2norm"
COMPRESSION_METHOD = CompressionMethod.PR_L2
RECOMMENDATION_METHOD = RecommendationMethod.SLAMP
RECOMMENDATION_RATIO = 0.6
OUTPUT_PATH = "./compressed_yolov5.pt"
compressed_model = compressor.recommendation_compression(
    model_id=model.model_id,
    model_name=COMPRESSED_MODEL_NAME,
    compression_method=COMPRESSION_METHOD,
    recommendation_method=RECOMMENDATION_METHOD,
    recommendation_ratio=RECOMMENDATION_RATIO,
    output_path=OUTPUT_PATH,
)
```

</details>

More commands can be found in the official NetsPresso Python Package Docs: https://nota-github.github.io/netspresso-python/build/html/index.html <br/>

Alternatively, you can do the same as above through the GUI on our website: https://console.netspresso.ai/models<br/><br/>

## 5. Fine-tuning the compressed Model</br>
Place the compressed model in the same place as the files obtained in Step 3('model_torchfx.pt', 'model_head.pt'). Change the compressed model name to 'model_compressed.pt'.
```bash
#Single GPU
python train.py --data coco.yaml --epochs 300 --weights model_compressed.pt --batch-size 128 --device 0 --netspresso

#Multi GPU
python -m torch.distributed.run --nproc_per_node 2 train.py --batch 64 --data coco.yaml --weights model_compressed.pt --device 0,1 --netspresso
```

Now you can use the compressed model however you like! </br></br>

## <div align="center">Contact</div>

Join our <a href="https://github.com/orgs/Nota-NetsPresso/discussions">Discussion Forum</a> for providing feedback or sharing your use cases, and if you want to talk more with Nota, please contact us <a href="https://www.nota.ai/contact-us">here</a>.</br>
Or you can also do it via email(contact@nota.ai) or phone(+82 2-555-8659)!

<br>
<div align="center">
  <a href="https://github.com/Nota-NetsPresso" style="text-decoration:none;">
    <img src="imgs/github.png" width="3%" alt="" /></a>
  <img src="imgs/logo-transparent.png" width="3%" alt="" />
  <a href="https://www.facebook.com/NotaAI" style="text-decoration:none;">
    <img src="imgs/facebook.png" width="3%" alt="" /></a>
  <img src="imgs/logo-transparent.png" width="3%" alt="" />
  <a href="https://twitter.com/nota_ai" style="text-decoration:none;">
    <img src="imgs/twitter.png" width="3%" alt="" /></a>
  <img src="imgs/logo-transparent.png" width="3%" alt="" />
  <a href="https://www.youtube.com/channel/UCeewYFAqb2EqwEXZCfH9DVQ" style="text-decoration:none;">
    <img src="imgs/youtube.png" width="3%" alt="" /></a>
  <img src="imgs/logo-transparent.png" width="3%" alt="" />
  <a href="https://www.linkedin.com/company/nota-incorporated" style="text-decoration:none;">
    <img src="imgs/linkedin.png" width="3%" alt="" /></a>
</div>