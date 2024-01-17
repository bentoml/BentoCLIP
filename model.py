import bentoml

# Save model to BentoML local model store
if __name__ == "__main__":
    try:
        bentoml.models.get("openai-clip")
        print("Model already exists")
    except:
        import huggingface_hub
        with bentoml.models.create(
            "openai-clip",
        ) as model_ref:
            huggingface_hub.snapshot_download("openai/clip-vit-base-patch32", local_dir=model_ref.path, local_dir_use_symlinks=False)