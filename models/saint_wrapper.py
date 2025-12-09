import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import logging
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
def attention(query, key, value, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    output = torch.matmul(p_attn, value)
    return output, p_attn
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, query, key, value):      
        nbatches = query.size(0)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
def intersample(query, key, value, dropout=None):
    "Calculate the intersample of a given query batch" 
    b, h, n, d = query.shape
    query, key, value = query.reshape(1, b, h, n*d), key.reshape(1, b, h, n*d), value.reshape(1, b, h, n*d)
    output, _ = attention(query, key, value)
    output = output.squeeze(0)
    output = output.reshape(b, h, n, d)
    return output
class MultiHeadedIntersampleAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedIntersampleAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, query, key, value):
        nbatches = query.size(0)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        x = intersample(query, key, value, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
    def forward(self, x):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        return self.sublayer[1](x, self.feed_forward)
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size) 
    def forward(self, x):
        "Pass the input through each layer in turn."
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
class SaintLayer(nn.Module):
    def __init__(self, msa, misa, size):
        super(SaintLayer, self).__init__()
        self.msa = msa
        self.misa = misa
        self.size = size
    def forward(self, x):
        return self.misa(self.msa(x))
class CategoricalEmbedding(nn.Module):
    def __init__(self, no_cat, embed_dim, feature=None):
        super(CategoricalEmbedding, self).__init__()
        self.embedding = nn.Embedding(no_cat, embed_dim)
        self.feature = feature
    def forward(self, x):
        return self.embedding(x)
class NumericalEmbedding(nn.Module):
    def __init__(self, embed_dim, feature=None):
        super(NumericalEmbedding, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU())
        self.feature = feature
    def forward(self, x):
        return self.linear(x)
class Embedding(nn.Module):
    def __init__(self, embed_dim, no_num, no_cat, cats):
        super(Embedding, self).__init__()
        assert no_cat == len(cats)
        self.embed_dim = embed_dim
        self.cat_embedding = nn.ModuleList()
        for cat in cats:
            self.cat_embedding.append(CategoricalEmbedding(cat, embed_dim))
        self.num_embedding = nn.ModuleList()
        for i in range(no_num):
            self.num_embedding.append(NumericalEmbedding(embed_dim))
        self.no_num = no_num
        self.no_cat = no_cat
    def forward(self, x):
        bs = x.shape[0]
        output = []
        for i, layer in enumerate(self.cat_embedding):
            output.append(layer(x[:, i].long()))
        for i, layer in enumerate(self.num_embedding):
            output.append(layer(x[:, self.no_cat + i].unsqueeze(1).float()))
        data = torch.stack(output, dim=1)
        return data
def make_saint(num_heads, embed_dim, num_layers, d_ff, dropout, dropout_ff=0.8):
    feed_forward = PositionwiseFeedForward(d_model=embed_dim, d_ff=d_ff, dropout=dropout_ff)
    self_attn = MultiHeadedAttention(num_heads, d_model=embed_dim)
    msa = EncoderLayer(embed_dim, self_attn, feed_forward, dropout)
    self_inter_attn = MultiHeadedIntersampleAttention(num_heads, d_model=embed_dim)
    misa = EncoderLayer(embed_dim, self_inter_attn, feed_forward, dropout)
    layer = SaintLayer(msa, misa, size=embed_dim)
    encoder = Encoder(layer, num_layers)
    for p in encoder.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return encoder
class ContrastiveLoss(nn.Module):
    def __init__(self, input_dim, proj_head_dim, temperature):
        super(ContrastiveLoss, self).__init__()
        self.projection_head_1 = nn.Sequential(
            nn.Linear(input_dim, proj_head_dim),
            nn.ReLU())
        self.projection_head_2 = nn.Sequential(
            nn.Linear(input_dim, proj_head_dim),
            nn.ReLU())
        self.temperature = temperature
    def contrastive_loss(self, zi, zi_prime):
        eps = 1e-7
        zi_product = torch.mm(zi, torch.t(zi_prime))
        zi_product = zi_product / self.temperature
        exp_zi_prod = torch.exp(zi_product)
        exp_zi_prod_sum = torch.sum(exp_zi_prod, dim=-1, keepdim=True)
        return -1.0 * torch.sum(torch.log(
            F.relu(torch.diag(exp_zi_prod / exp_zi_prod_sum)) + eps))
    def forward(self, ri, ri_prime):
        ri = ri.reshape(ri.shape[0], -1)
        ri_prime = ri_prime.reshape(ri_prime.shape[0], -1)
        zi = self.projection_head_1(ri)
        zi_prime = self.projection_head_2(ri_prime)
        return self.contrastive_loss(zi, zi_prime)
class DenoisingLoss(nn.Module):
    def __init__(self, no_num, no_cat, cats, input_dim):
        super(DenoisingLoss, self).__init__()
        self.no_num = no_num
        self.no_cat = no_cat
        self.cats = cats
        self.cat_mlps = nn.ModuleList()
        for i in range(1, self.no_cat):
            self.cat_mlps.append(nn.Linear(input_dim, self.cats[i]))
        num_mlp = nn.Sequential(nn.Linear(input_dim, 1), nn.ReLU())
        self.num_mlps = clones(num_mlp, no_num)
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
    def forward(self, ri_prime, xi):
        denoising_loss = 0.0
        num_loss = 0.0
        cat_loss = 0.0
        for feat_idx in range(1, self.no_cat):
            ri_feat = self.cat_mlps[feat_idx - 1](ri_prime[:, feat_idx, :].squeeze())
            xi_feat = xi[:, feat_idx]
            cat_loss += self.ce(ri_feat.float(), xi_feat.long())
        for feat_idx in range(self.no_num):
            idx = self.no_cat + feat_idx
            ri_feat = self.num_mlps[feat_idx](ri_prime[:, idx, :])
            xi_feat = xi[:, idx]
            num_loss += self.mse(ri_feat.squeeze().float(), xi_feat.float())
        denoising_loss = num_loss + cat_loss
        return denoising_loss
class CutMix:
    def __init__(self, prob_cutmix):
        super(CutMix, self).__init__()
        self.prob_cutmix = prob_cutmix
    def __call__(self, x_i):
        shuffled_index = torch.randperm(x_i.shape[0])
        x_a = x_i[shuffled_index]
        prob_matrix = torch.ones(x_i.shape) * (1 - self.prob_cutmix)
        m_binary_matrix = torch.empty_like(x_i)
        torch.bernoulli(prob_matrix, out=m_binary_matrix)
        xi_cutmix = m_binary_matrix * x_i + (1 - m_binary_matrix) * x_a
        return xi_cutmix
class Mixup:
    def __init__(self, alpha):
        super(Mixup, self).__init__()
        self.alpha = alpha
    def __call__(self, xi_embed):
        shuffled_index = torch.randperm(xi_embed.shape[0])
        xb_prime = xi_embed[shuffled_index]
        p_i = self.alpha * xi_embed + (1 - self.alpha) * xb_prime
        return p_i
class SAINT(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, d_ff, dropout, 
                 no_num, no_cat, cats, num_classes, cls_token_idx=0):
        super(SAINT, self).__init__()
        self.embedding = Embedding(embed_dim, no_num, no_cat, cats)
        self.transformer = make_saint(num_heads, embed_dim, num_layers, d_ff, dropout)
        self.cls_token_idx = cls_token_idx
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
    def forward(self, x):
        embedded = self.embedding(x)
        encoded = self.transformer(embedded)
        if self.cls_token_idx < encoded.shape[1]:
            pooled = encoded[:, self.cls_token_idx, :]
        else:
            pooled = encoded.mean(dim=1)
        output = self.classifier(pooled)
        return output
class SAINTClassifierWrapper:
    def __init__(self, logger=None, device='auto', batch_size=256, num_epochs=100,
                 learning_rate=1e-3, weight_decay=1e-5, dropout=0.1, 
                 hidden_dim=256, num_layers=4, num_heads=8, d_token=64,
                 seed=42):
        self.model = None
        self.logger = logger or logging.getLogger(__name__)
        self._estimator_type = "classifier"
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_token = d_token
        self.seed = seed
    def fit(self, X_train, y_train, params=None):
        if params is None:
            params = {
                'device': self.device,
                'batch_size': self.batch_size,
                'num_epochs': self.num_epochs,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'dropout': self.dropout,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'num_heads': self.num_heads,
                'd_token': self.d_token,
                'seed': self.seed
            }
        try:
            X_train_np = np.array(X_train)
            y_train_np = np.array(y_train)
            n_features = X_train_np.shape[1]
            n_classes = len(np.unique(y_train_np))
            self.model = SAINT(
                embed_dim=params['hidden_dim'],
                num_heads=params['num_heads'],
                num_layers=params['num_layers'],
                d_ff=params['hidden_dim'] * 4,
                dropout=params['dropout'],
                no_num=n_features,
                no_cat=0,
                cats=[],
                num_classes=n_classes,
                cls_token_idx=0
            )
            try:
                device = torch.device('cuda' if torch.cuda.is_available() and params['device'] != 'cpu' else 'cpu')
                self.model.to(device)
            except Exception as dev_e:
                self.logger.warning(f"CUDA device initialization failed, falling back to CPU: {str(dev_e)}")
                device = torch.device('cpu')
                self.model.to(device)
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=params['learning_rate'],
                weight_decay=params['weight_decay']
            )
            if n_classes > 1:
                criterion = torch.nn.CrossEntropyLoss()
            else:
                criterion = torch.nn.BCEWithLogitsLoss()
            self.model.train()
            for epoch in range(params['num_epochs']):
                for i in range(0, len(X_train_np), params['batch_size']):
                    batch_X = torch.tensor(X_train_np[i:i+params['batch_size']], dtype=torch.float32).to(device)
                    batch_y = torch.tensor(y_train_np[i:i+params['batch_size']], dtype=torch.long if n_classes > 1 else torch.float32).to(device)
                    optimizer.zero_grad()
                    try:
                        output = self.model(batch_X)
                    except Exception as fwd_e:
                        self.logger.warning(f"SAINT forward failed, retrying with CPU fallback: {str(fwd_e)}")
                        cpu_device = torch.device('cpu')
                        self.model.to(cpu_device)
                        batch_X = batch_X.to(cpu_device)
                        output = self.model(batch_X)
                    if n_classes > 1:
                        loss = criterion(output, batch_y)
                    else:
                        loss = criterion(output.squeeze(), batch_y.float())
                    loss.backward()
                    optimizer.step()
            self.model.eval()
            with torch.no_grad():
                X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32).to(device)
                outputs = self.model(X_train_tensor)
                if n_classes > 1:
                    train_pred = torch.argmax(outputs, dim=1).cpu().numpy()
                    train_probs = torch.softmax(outputs, dim=1).cpu().numpy()
                    train_auc = roc_auc_score(y_train_np, train_probs[:, 1] if train_probs.shape[1] > 1 else train_probs[:, 0])
                else:
                    train_pred = (outputs > 0).cpu().numpy().astype(int)
                    train_probs_sigmoid = torch.sigmoid(outputs).cpu().numpy()
                    train_probs = np.column_stack([1 - train_probs_sigmoid, train_probs_sigmoid])
                    train_auc = roc_auc_score(y_train_np, train_probs[:, 1])
                train_acc = accuracy_score(y_train_np, train_pred)
            self.logger.info(f"SAINT Classifier training complete (params: {params}) - Train AUC: {train_auc:.4f}, Train accuracy: {train_acc:.4f}")
            return True
        except Exception as e:
            self.logger.error(f"SAINT Classifier training failed: {str(e)}")
            return False
    def predict(self, X_test):
        if self.model is None:
            return np.zeros(len(X_test))
        try:
            X_test_np = np.array(X_test)
            device = next(self.model.parameters()).device
            self.model.eval()
            predictions = []
            with torch.no_grad():
                for i in range(0, len(X_test_np), self.batch_size):
                    batch_X = torch.tensor(X_test_np[i:i+self.batch_size], dtype=torch.float32).to(device)
                    output = self.model(batch_X)
                    if output.shape[-1] > 1:
                        pred = torch.argmax(output, dim=1).cpu().numpy()
                    else:
                        pred = (output > 0).cpu().numpy().astype(int)
                    predictions.extend(pred)
            return np.array(predictions)
        except Exception as e:
            self.logger.error(f"SAINT Classifier prediction failed: {str(e)}")
            return np.zeros(len(X_test))
    def predict_proba(self, X_test):
        if self.model is None:
            return np.zeros((len(X_test), 2))
        try:
            X_test_np = np.array(X_test)
            device = next(self.model.parameters()).device
            self.model.eval()
            probabilities = []
            with torch.no_grad():
                for i in range(0, len(X_test_np), self.batch_size):
                    batch_X = torch.tensor(X_test_np[i:i+self.batch_size], dtype=torch.float32).to(device)
                    output = self.model(batch_X)
                    if output.shape[-1] > 1:
                        prob = torch.softmax(output, dim=1).cpu().numpy()
                    else:
                        if output.dim() == 1:
                            prob_sigmoid = torch.sigmoid(output).cpu().numpy()
                            prob = np.column_stack([1 - prob_sigmoid, prob_sigmoid])
                        else:
                            prob_sigmoid = torch.sigmoid(output).cpu().numpy()
                            prob = np.column_stack([1 - prob_sigmoid, prob_sigmoid])
                    probabilities.extend(prob)
            return np.array(probabilities)
        except Exception as e:
            self.logger.error(f"SAINT Classifier probability prediction failed: {str(e)}")
            return np.zeros((len(X_test), 2))
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        metrics = {}
        try:
            if len(np.unique(y_test)) == 2:
                metrics['auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                metrics['auc'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            metrics['f1'] = f1_score(y_test, y_pred, average='weighted')
            metrics['precision'] = precision_score(y_test, y_pred, average='weighted')
            metrics['recall'] = recall_score(y_test, y_pred, average='weighted')
        except:
            metrics = {key: 0.0 for key in ['auc', 'accuracy', 'f1', 'precision', 'recall']}
        return metrics, y_pred, y_pred_proba
    def get_params(self, deep=True):
        return {
            'device': self.device,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'dropout': self.dropout,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'd_token': self.d_token,
            'seed': self.seed
        }
    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
    def score(self, X, y):
        y_pred = self.predict(X)
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, y_pred) 