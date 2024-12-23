<div align="center">
    <h1 align="center">Serving CLIP with BentoML</h1>
</div>

[CLIP (Contrastive Language–Image Pre-training)](https://openai.com/research/clip) is a machine learning model developed by OpenAI. It is versatile and excels in tasks like zero-shot learning, image classification, and image-text matching without needing specific training for each task.

This is a BentoML example project, demonstrating how to build a CLIP inference API server, using the [clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) model. See [here](https://docs.bentoml.com/en/latest/examples/overview.html) for a full list of BentoML example projects.

## Install dependencies

```bash
git clone https://github.com/bentoml/BentoClip.git
cd BentoClip

# Recommend Python 3.11
pip install -r requirements.txt
```

## Run the BentoML Service

We have defined a BentoML Service in `service.py`. Run `bentoml serve` in your project directory to start the Service.

```bash
$ bentoml serve .

2024-01-08T09:07:28+0000 [INFO] [cli] Prometheus metrics for HTTP BentoServer from "service:CLIPService" can be accessed at http://localhost:3000/metrics.
2024-01-08T09:07:28+0000 [INFO] [cli] Starting production HTTP BentoServer from "service:CLIPService" listening on http://localhost:3000 (Press CTRL+C to quit)
Model clip loaded device: cuda
```

The Service is accessible at [http://localhost:3000](http://localhost:3000/). You can interact with it using the Swagger UI or in other different ways:

CURL

```bash
curl -s \
     -X POST \
     -F 'items=@demo.jpg' \
     http://localhost:3000/encode_image
```

Python client

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

For detailed explanations of the Service code, see [CLIP embeddings](https://docs.bentoml.org/en/latest/use-cases/embeddings/clip-embeddings.html).

## Deploy to BentoCloud

After the Service is ready, you can deploy the application to BentoCloud for better management and scalability. [Sign up](https://www.bentoml.com/) if you haven't got a BentoCloud account.

Make sure you have [logged in to BentoCloud](https://docs.bentoml.com/en/latest/bentocloud/how-tos/manage-access-token.html), then run the following command to deploy it.

```bash
bentoml deploy .
```

Once the application is up and running on BentoCloud, you can access it via the exposed URL.

**Note**: For custom deployment in your own infrastructure, use [BentoML to generate an OCI-compliant image](https://docs.bentoml.com/en/latest/guides/containerization.html).
