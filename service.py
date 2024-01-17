import bentoml
from PIL.Image import Image
import numpy as np
from typing import List
from pydantic import Field


@bentoml.service()
class CLIPService:
    model_ref = bentoml.models.get("openai-clip")
    
    def __init__(self) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(self.model_ref.path).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_ref.path)
        self.logit_scale = self.model.logit_scale.item() if self.model.logit_scale.item() else 4.60517
        print("Model clip loaded", "device:", self.device)

    @bentoml.api(batchable=True)
    async def encode_image(self, items: List[Image]) -> np.ndarray:
        '''
        generate the 512-d embeddings of the images
        '''
        inputs = self.processor(images=items, return_tensors="pt", padding=True).to(self.device)
        image_embeddings = self.model.get_image_features(**inputs)
        return image_embeddings.cpu().detach().numpy()
    

    @bentoml.api(batchable=True)
    async def encode_text(self, items: List[str]) -> np.ndarray:
        '''
        generate the 512-d embeddings of the texts
        '''
        inputs = self.processor(text=items, return_tensors="pt", padding=True).to(self.device)
        text_embeddings = self.model.get_text_features(**inputs)
        return text_embeddings.cpu().detach().numpy()
    
    @bentoml.api
    async def rank(self, queries: List[Image],  candidates : List[str] = Field(["picture of a dog", "picture of a cat"], description="list of description candidates")) -> dict[str, List[List[float]]]:
        '''
        return the similarity between the query images and the candidate texts
        '''
        # Encode embeddings
        query_embeds = await self.encode_image(queries)
        candidate_embeds = await self.encode_text(candidates)

        # Compute cosine similarities
        cosine_similarities = self.cosine_similarity(query_embeds, candidate_embeds)
        logit_scale = np.exp(self.logit_scale)
        # Compute softmax scores
        prob_scores = self.softmax(logit_scale * cosine_similarities)
        return {
            "probabilities": prob_scores.tolist(),
            "cosine_similarities" : cosine_similarities.tolist(),
        }
        
    @staticmethod
    def cosine_similarity(query_embeds, candidates_embeds):
        # Normalize each embedding to a unit vector
        query_embeds /= np.linalg.norm(query_embeds, axis=1, keepdims=True)
        candidates_embeds /= np.linalg.norm(candidates_embeds, axis=1, keepdims=True)

        # Compute cosine similarity
        cosine_similarities = np.matmul(query_embeds, candidates_embeds.T)

        return cosine_similarities
    
    @staticmethod
    def softmax(scores):
        # Compute softmax scores (probabilities)
        exp_scores = np.exp(
            scores - np.max(scores, axis=-1, keepdims=True)
        )  # Subtract max for numerical stability
        return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    

if __name__ == "__main__":
    CLIPService.serve_http()