import os
import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModel


class E5CrossEncoder(nn.Module):
    """
    Cross-encoder built on top of a transformer encoder.

    The model encodes a concatenated input of:
        "query: ... passage: ..."

    and produces a single relevance score per input pair.

    Architecture:
        Transformer encoder → mean pooling → MLP scoring head
    """

    def __init__(self, encoder):
        super().__init__()

        self.encoder = encoder
        hidden_size = encoder.config.hidden_size

        # self.scoring = nn.Linear(hidden_size, 1)
        self.scoring = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2),
                                     nn.ReLU(),
                                     nn.Dropout(0.1),
                                     nn.Linear(hidden_size // 2, hidden_size // 4),
                                     nn.ReLU(),
                                     nn.Dropout(0.1),
                                     nn.Linear(hidden_size // 4, 1))

    def forward(self, input_ids, attention_mask):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # token_embeddings: [B, T, H]
        token_embeddings = outputs.last_hidden_state

        # expanded_mask: [B, T, 1]
        expanded_mask = attention_mask.unsqueeze(-1).to(dtype=token_embeddings.dtype)

        # numerator: [B, H]
        numerator = (token_embeddings * expanded_mask).sum(dim=1)

        # denominator: [B, 1]
        denominator = expanded_mask.sum(dim=1).clamp(min=1e-9)

        # pooled: [B, H]
        pooled = numerator / denominator

        # scores: [B]
        scores = self.scoring(pooled).squeeze(-1)

        return scores


class E5Reranker:
    """
    Cross-encoder reranker based on multilingual-e5-base.

    The model receives a query–passage pair and predicts a relevance score,
    allowing the retrieved passages to be reranked before being passed to the LLM.
    """

    def __init__(self, model_name="intfloat/multilingual-e5-base", device=None):

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        encoder = AutoModel.from_pretrained(model_name)
        self.model = E5CrossEncoder(encoder)

        # Device setup
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def build_inputs(self, queries, passages):
        """
        Build the tokenized batch of query–passage pairs for the cross-encoder.

        Each query is paired with the corresponding passage to form a single
        input sequence of the form:

            "query: {query} passage: {passage}"

        The resulting batch is tokenized and returned as PyTorch tensors ready
        to be fed into the transformer model.

        Args:
            queries (List[str]): List of queries.
            passages (List[str]): List of passages paired with the queries.

        Returns:
            BatchEncoding: Tokenized batch containing the transformer inputs
            (e.g., input_ids and attention_mask).
        """

        texts = [
            f"query: {q} passage: {p}"
            for q, p in zip(queries, passages)
        ]

        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

    @torch.no_grad()
    def rerank(self, query, passages, top_k=None):
        """
        Rank the candidate passages according to their relevance to the query.

        Args:
            query (str): User query.
            passages (List[str]): Candidate passages to rerank.
            top_k (int, optional): Number of highest-ranked passages to return.
                If None, all passages are returned.

        Returns:
            List[Tuple[str, float]]: Ranked (passage, score) pairs sorted in
            descending order of relevance.
        """        
        
        if not passages:
            return []

        self.model.eval()

        queries = [query] * len(passages)

        inputs = self.build_inputs(queries, passages)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        scores = self.model(**inputs).cpu()

        ranked_idx = torch.argsort(scores, descending=True)

        if top_k is not None:
            ranked_idx = ranked_idx[:top_k]

        return [
            (passages[i], scores[i].item())
            for i in ranked_idx.tolist()
        ]


    def save(self, path):
        """
        Save the reranker model, tokenizer and configuration to disk.
        """        

        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, "model.pt"))
        self.tokenizer.save_pretrained(path)
        with open(os.path.join(path, "config.txt"), "w") as f:
            f.write(self.model_name)   

    @staticmethod
    def load(path, device=None):
        """
        Load a previously saved reranker from disk.
        """        

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        with open(os.path.join(path, "config.txt")) as f:
            model_name = f.read().strip()        

        tokenizer = AutoTokenizer.from_pretrained(path)

        encoder = AutoModel.from_pretrained(model_name)
        model = E5CrossEncoder(encoder)
       
        state_dict = torch.load(
            os.path.join(path, "model.pt"),
            map_location=device
        )
        
        model.load_state_dict(state_dict)        
        model.to(device)
        model.eval()

        reranker = E5Reranker.__new__(E5Reranker)
        reranker.model = model
        reranker.tokenizer = tokenizer
        reranker.model_name = model_name
        reranker.device = device

        return reranker
