CLIP (Contrastive Language–Image Pre-training) is a machine learning model developed by OpenAI. It is versatile and excels in tasks like zero-shot learning, image classification, and image-text matching without needing specific training for each task. This makes it ideal for a wide range of applications, including content recommendation, image captioning, visual search, and automated content moderation.

This document demonstrates how to build a CLIP application using BentoML, powered by the [clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) model.

## **Prerequisites**

- You have installed Python 3.8+ and `pip`. See the [Python downloads page](https://www.python.org/downloads/) to learn more.
- You have a basic understanding of key concepts in BentoML, such as Services. We recommend you read [Quickstart](https://docs.bentoml.com/en/latest/get-started/quickstart.html) first.
- (Optional) We recommend you create a virtual environment for dependency isolation for this project. See Installation for details.

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run the BentoML Service

We have defined a BentoML Service in `service.py`. Run `bentoml serve` in your project directory to start the Service.

```python
$ bentoml serve .

2024-01-08T09:07:28+0000 [INFO] [cli] Prometheus metrics for HTTP BentoServer from "service:CLIPService" can be accessed at http://localhost:3000/metrics.
2024-01-08T09:07:28+0000 [INFO] [cli] Starting production HTTP BentoServer from "service:CLIPService" listening on http://localhost:3000 (Press CTRL+C to quit)
Model clip loaded device: cuda
```

Access the Service UI at [http://0.0.0.0:3000](http://0.0.0.0:3000/). You can interact with it using Swagger UI or in other different ways:

CURL

```bash
curl -s \
     -X POST \
     -F 'items=@demo.jpg' \
     http://localhost:3000/encode_image
```

Python Client

```python
import bentoml
from pathlib import Path

with bentoml.SyncHTTPClient("http://localhost:3000") as client:
    result = client.encode_image(
        items=[
            Path("demo.jpg"),
        ],
    )
```

## Deploy the application to BentoCloud

After the Service is ready, you can deploy the application to BentoCloud for better management and scalability. A YAML configuration file (`bentofile.yaml`) is used to define the build options and package your application into a Bento. See [Bento build options](https://docs.bentoml.com/en/latest/concepts/bento.html#bento-build-options) to learn more.

Make sure you have logged in to BentoCloud, then run the following command in your project directory to deploy to BentoCloud. Under the hood, this command builds a Bento, pushes, and deploys it on BentoCloud.

```bash
bentoml deploy .
```

**Note**: Alternatively, you can manually build the Bento, containerize the Bento as a Docker image, and deploy it in any Docker-compatible environment. See [Docker deployment](https://docs.bentoml.org/en/latest/concepts/deploy.html#docker) for details.

Once the application is up and running on BentoCloud, you can access it via the exposed URL.
