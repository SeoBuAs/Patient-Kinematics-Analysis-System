import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from torch.nn import Module, ModuleList
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from einops import rearrange, repeat
import logging
try:
    from hyper_connections import HyperConnections
except ImportError:
    print("Warning: hyper-connections not installed. Using simplified version.")
    class HyperConnections:
        @staticmethod
        def get_init_and_expand_reduce_stream_functions(num_residual_streams, disable=False):
            if disable or num_residual_streams == 1:
                def init_hyper_conn(dim, branch):
                    return branch
                def expand_streams(x):
                    return x
                def reduce_streams(x):
                    return x
                return init_hyper_conn, expand_streams, reduce_streams
            else:
                def init_hyper_conn(dim, branch):
                    return branch
                def expand_streams(x):
                    return x
                def reduce_streams(x):
                    return x
                return init_hyper_conn, expand_streams, reduce_streams
class GEGLU(Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)
def FeedForward(dim, mult=4, dropout=0.):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )
class Attention(Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        h = self.heads
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        dropped_attn = self.dropout(attn)
        out = einsum('b h i j, b h j d -> b h i d', dropped_attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        out = self.to_out(out)
        return out, attn
class Transformer(Module):
    def __init__(self, dim, depth, heads, dim_head, attn_dropout, ff_dropout, num_residual_streams=4):
        super().__init__()
        init_hyper_conn, self.expand_streams, self.reduce_streams = HyperConnections.get_init_and_expand_reduce_stream_functions(
            num_residual_streams, disable=num_residual_streams == 1
        )
        self.layers = ModuleList([])
        for _ in range(depth):
            self.layers.append(ModuleList([
                init_hyper_conn(dim=dim, branch=Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout)),
                init_hyper_conn(dim=dim, branch=FeedForward(dim, dropout=ff_dropout)),
            ]))
    def forward(self, x, return_attn=False):
        post_softmax_attns = []
        x = self.expand_streams(x)
        for attn, ff in self.layers:
            x, post_softmax_attn = attn(x)
            post_softmax_attns.append(post_softmax_attn)
            x = ff(x)
        x = self.reduce_streams(x)
        if not return_attn:
            return x
        return x, torch.stack(post_softmax_attns)
class NumericalEmbedder(Module):
    def __init__(self, dim, num_numerical_types):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_numerical_types, dim))
        self.biases = nn.Parameter(torch.randn(num_numerical_types, dim))
    def forward(self, x):
        x = rearrange(x, 'b n -> b n 1')
        return x * self.weights + self.biases
class FTTransformer(Module):
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head=16,
        dim_out=1,
        num_special_tokens=2,
        attn_dropout=0.,
        ff_dropout=0.,
        num_residual_streams=4
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert len(categories) + num_continuous > 0, 'input shape must not be null'
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)
        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens
        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=num_special_tokens)
            categories_offset = categories_offset.cumsum(dim=-1)[:-1]
            self.register_buffer('categories_offset', categories_offset)
            self.categorical_embeds = nn.Embedding(total_tokens, dim)
        self.num_continuous = num_continuous
        if self.num_continuous > 0:
            self.numerical_embedder = NumericalEmbedder(dim, self.num_continuous)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            num_residual_streams=num_residual_streams
        )
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim_out)
        )
    def forward(self, x_categ, x_numer, return_attn=False):
        assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'
        xs = []
        if self.num_unique_categories > 0:
            x_categ = x_categ + self.categories_offset
            x_categ = self.categorical_embeds(x_categ)
            xs.append(x_categ)
        if self.num_continuous > 0:
            x_numer = self.numerical_embedder(x_numer)
            xs.append(x_numer)
        x = torch.cat(xs, dim=1)
        b = x.shape[0]
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x, attns = self.transformer(x, return_attn=True)
        x = x[:, 0]
        logits = self.to_logits(x)
        if not return_attn:
            return logits
        return logits, attns
class FTTransformerWrapper:
    def __init__(self, logger=None, dim=128, depth=3, heads=4, dim_head=32, 
                 attn_dropout=0.1, ff_dropout=0.1, num_residual_streams=4, 
                 batch_size=32, epochs=50, lr=1e-3, early_stopping_patience=10,
                 seed: int = 42):
        self.logger = logger
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.num_residual_streams = num_residual_streams
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.early_stopping_patience = early_stopping_patience
        self.seed = seed
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_fitted = False
    def fit(self, X_train, y_train, params=None):
        if params:
            for key, value in params.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        try:
            import random as _random
            _random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
        except Exception:
            pass
        if self.logger:
            self.logger.info(f"FT-Transformer training started (device: {self.device})")
            self.logger.info(f"Hyperparameters: dim={self.dim}, depth={self.depth}, heads={self.heads}")
        X_train_tensor = torch.FloatTensor(X_train.values).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train.values).to(self.device)
        num_continuous = X_train.shape[1]
        categories = []
        self.model = FTTransformer(
            categories=categories,
            num_continuous=num_continuous,
            dim=self.dim,
            depth=self.depth,
            heads=self.heads,
            dim_head=self.dim_head,
            dim_out=1,
            attn_dropout=self.attn_dropout,
            ff_dropout=self.ff_dropout,
            num_residual_streams=self.num_residual_streams
        ).to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        best_loss = float('inf')
        patience_counter = 0
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            num_batches = 0
            for i in range(0, len(X_train_tensor), self.batch_size):
                batch_X = X_train_tensor[i:i+self.batch_size]
                batch_y = y_train_tensor[i:i+self.batch_size]
                optimizer.zero_grad()
                x_categ = torch.zeros(batch_X.shape[0], 0, dtype=torch.long).to(self.device)
                x_numer = batch_X
                outputs = self.model(x_categ, x_numer)
                outputs = outputs.view(-1)
                loss = criterion(outputs, batch_y.view(-1).float())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
            avg_loss = total_loss / num_batches
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if self.logger and epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
            if patience_counter >= self.early_stopping_patience:
                if self.logger:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        self.is_fitted = True
        if self.logger:
            self.logger.info("FT-Transformer training complete")
        return self
    def predict(self, X_test):
        if not self.is_fitted:
            raise ValueError("Model not trained.")
        self.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test.values).to(self.device)
            x_categ = torch.zeros(X_test_tensor.shape[0], 0, dtype=torch.long).to(self.device)
            x_numer = X_test_tensor
            outputs = self.model(x_categ, x_numer)
            predictions = torch.sigmoid(outputs.view(-1))
            return (predictions > 0.5).float().cpu().numpy()
    def predict_proba(self, X_test):
        if not self.is_fitted:
            raise ValueError("Model not trained.")
        self.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test.values).to(self.device)
            x_categ = torch.zeros(X_test_tensor.shape[0], 0, dtype=torch.long).to(self.device)
            x_numer = X_test_tensor
            outputs = self.model(x_categ, x_numer)
            probabilities = torch.sigmoid(outputs.view(-1))
            proba = torch.stack([1 - probabilities, probabilities], dim=1).cpu().numpy()
            return proba
    def evaluate(self, X_test, y_test):
        if not self.is_fitted:
            raise ValueError("Model not trained.")
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        y_pred = self.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }
        if self.logger:
            self.logger.info(f"Evaluation results: {metrics}")
        return metrics
    def get_params(self, deep=True):
        return {
            'dim': self.dim,
            'depth': self.depth,
            'heads': self.heads,
            'dim_head': self.dim_head,
            'attn_dropout': self.attn_dropout,
            'ff_dropout': self.ff_dropout,
            'num_residual_streams': self.num_residual_streams,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'lr': self.lr,
            'early_stopping_patience': self.early_stopping_patience
        }
    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
    def score(self, X, y):
        if not self.is_fitted:
            raise ValueError("Model not trained.")
        y_pred = self.predict(X)
        return f1_score(y, y_pred) 