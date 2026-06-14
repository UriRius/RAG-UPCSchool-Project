import torch
from transformers import AutoModel, AutoTokenizer


class E5Embedder:
    """E5 embeddings with query:/passage: prefixes and CLS token pooling."""

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, *, local_files_only: bool = False):
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, local_files_only=local_files_only
        )
        model = AutoModel.from_pretrained(
            model_name_or_path, local_files_only=local_files_only
        )
        return cls(tokenizer, model)

    def embed_query(self, text: str) -> list[float]:
        prefixed = f"query: {text}"
        inputs = self.tokenizer(prefixed, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings[0].numpy().tolist()

    def embed_passages(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = [f"passage: {t}" for t in texts[i : i + batch_size]]
            inputs = self.tokenizer(batch, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.extend(embeddings.numpy().tolist())
        return all_embeddings
