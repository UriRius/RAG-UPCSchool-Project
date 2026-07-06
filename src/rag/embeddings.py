import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from rag.config import EMBEDDING_STYLE


def mean_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class E5Embedder:
    """E5 embeddings — estilo Tristan (mean pool, sin prefijo) o e5 (CLS + query:/passage:)."""

    def __init__(self, tokenizer, model, *, style: str = EMBEDDING_STYLE):
        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()
        self.style = style.lower()

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        local_files_only: bool = False,
        style: str | None = None,
    ):
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, local_files_only=local_files_only
        )
        model = AutoModel.from_pretrained(
            model_name_or_path, local_files_only=local_files_only
        )
        return cls(tokenizer, model, style=style or EMBEDDING_STYLE)

    def _encode_texts(self, texts: list[str]) -> list[list[float]]:
        all_embeddings: list[list[float]] = []
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
                if self.style == "tristan":
                    embeddings = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
                else:
                    embeddings = outputs.last_hidden_state[:, 0, :]
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                all_embeddings.extend(embeddings.cpu().tolist())
        return all_embeddings

    def _encode_one(self, text: str) -> list[float]:
        if self.style == "e5":
            text = f"query: {text}"
        return self._encode_texts([text])[0]

    def embed_query(self, text: str) -> list[float]:
        return self._encode_one(text)

    def embed_passages(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        if self.style == "e5":
            texts = [f"passage: {t}" for t in texts]
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            all_embeddings.extend(self._encode_texts(texts[i : i + batch_size]))
        return all_embeddings
