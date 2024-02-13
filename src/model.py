import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class POIEmbeddings(nn.Module):
    def __init__(self, poi_size, poi_embed_dim):
        super(POIEmbeddings, self).__init__()
        self.emb = nn.Embedding(poi_size, poi_embed_dim)

    def forward(self, traj):
        x = self.emb(traj)
        return x


class TimeEmbeddings(nn.Module):
    def __init__(self, time_size, time_embed_dim):
        super(TimeEmbeddings, self).__init__()
        self.emb = nn.Embedding(time_size, time_embed_dim)

    def forward(self, time):
        x = self.emb(time)
        return x

class FuseEmbeddings(nn.Module):
    def __init__(self, time_embed_dim, poi_embed_dim):
        super(FuseEmbeddings, self).__init__()
        embed_dim = time_embed_dim + poi_embed_dim
        self.fuse_embed = nn.Linear(embed_dim, embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, time_embed, poi_embed):
        x = self.fuse_embed(torch.cat([time_embed, poi_embed], dim=2))
        x = self.leaky_relu(x)
        return x


# ============== Time Encoding ================== #
def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0

    return torch.cat([v1, v2], 1).squeeze()


class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class Time2Vec(nn.Module):
    def __init__(self, activation, out_dim):
        super(Time2Vec, self).__init__()
        self.out_dim = out_dim
        if activation == "sin":
            self.l1 = SineActivation(1, out_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(1, out_dim)

    def forward(self, x):

        new_x = torch.zeros(x.shape[0], x.shape[1], self.out_dim)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                new_x[i][j] = self.l1(x[i][j].unsqueeze(0))  # [b,l,d]
        return new_x.cuda()


# =================Transformer framework================== #
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, embed_size, nhead, nhid, nlayers, dropout=0.3):
        super(TransformerModel, self).__init__()

        # self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embed_size = embed_size

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, src_mask):
        # src = src * math.sqrt(self.embed_size)
        # src = self.pos_encoder(src)
        x = self.transformer_encoder(src)

        return x


# =============================Recommender========================= #
class Recommender(nn.Module):
    def __init__(self, out_dim, poi_size):
        super(Recommender, self).__init__()
        self.fc = nn.Linear(out_dim, poi_size)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, outputs):
        x = self.fc(outputs)
        x = self.leaky_relu(x)
        return x


# ============================Penalty=============================== #
class Drifting(nn.Module):
    def __init__(self, beta):
        super(Drifting, self).__init__()
        self.beta = beta

    def forward(self, fix_outputs):

        batch_size, seq_len, _ = fix_outputs.size()
        max_num_moves = seq_len - 1
        total_similarity = 0

        for num_moves in range(1, max_num_moves + 1):
            shift_outputs = fix_outputs[:, num_moves:]
            similarity = F.cosine_similarity(fix_outputs[:, :-num_moves], shift_outputs, dim=-1)
            total_similarity += similarity.mean()
        avg_similarity = total_similarity / (seq_len - 1)
        repetition_penalty_loss = -torch.log(1 - 0.5 * (avg_similarity.mean() + 1)) * self.beta

        return repetition_penalty_loss


class Guiding(nn.Module):
    def __init__(self, out_dim, poi_size):
        super(Guiding, self).__init__()
        self.predictor = Recommender(out_dim, poi_size)
        # self.confidence = nn.Linear(poi_size, 1)

    def forward(self, outputs, AM, PM):
        fix_outputs = self.predictor(outputs)  # [b,l,d] -> [b,l,v]
        clipped_PM = PM[:, :fix_outputs.shape[1]]  # [v,l_max] -> [v,l]
        clipped_outputs = fix_outputs * (clipped_PM.T.unsqueeze(0).expand(fix_outputs.shape[0], -1, -1))  # [b,l,v]

        return clipped_outputs

# Construct total framework(AR-Trip)
class ARModel(nn.Module):
    def __init__(self, venue_size, hour_size, drifting, guiding, beta=0.1,
                 max_length_venue_id=100, d_model=128, n_head=4, num_encoder_layers=1):

        super(ARModel, self).__init__()
        # initial hyperparameter
        venue_dim = d_model
        hour_dim = d_model
        position_dim = d_model
        self.hour_size = hour_size
        self.drifting = drifting
        self.guiding = guiding
        # model setting
        self.poi_emb = POIEmbeddings(venue_size, venue_dim)
        self.hour_emb = Time2Vec('cos', hour_dim)
        self.merge_emb = FuseEmbeddings(hour_dim, venue_dim)
        self.pos_emb = nn.Embedding(max_length_venue_id, position_dim)
        # build the relative structure
        self.transformer_encoder = TransformerModel(embed_size=venue_dim + hour_dim + position_dim, nhead=n_head,
                                                    nhid=2048, nlayers=num_encoder_layers)

        self.predictor = Recommender(venue_dim + hour_dim + position_dim, venue_size)
        self.drift = Drifting(beta)
        self.guide = Guiding(venue_dim + hour_dim + position_dim, venue_size)

    def forward(self, poi_input, hour_input, am, pm, src_mask):

        hour_input = torch.gather(torch.linspace(start=0, end=1, steps=self.hour_size).cuda(),
                                  0, hour_input.flatten()).reshape(hour_input.shape)  # [b,l]
        batch_size, seq_length = poi_input.size()
        # initialize
        poi_embedded = self.poi_emb(poi_input)  # [b,l,d]
        hour_embedded = self.hour_emb(hour_input)  # [b,d] -> [b,l,d]
        # Generate position embeddings for each position in the sequence
        position_ids = torch.arange(seq_length, dtype=torch.long, device=poi_input.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embedded = self.pos_emb(position_ids)
        # the transformer src
        merge_embedded = self.merge_emb(hour_embedded, poi_embedded)
        model_input = torch.cat([merge_embedded, position_embedded], dim=2)  # [b,l,3d]
        model_output = self.transformer_encoder(model_input, src_mask)
        # guiding mechanism
        if self.guiding:
            clipped_output = self.guide(model_output, am, pm)
            poi_output = self.predictor(model_output)
            poi_output = poi_output + clipped_output

        else:
            poi_output = self.predictor(model_output)  # [b,l,v]
        # drifting mechanism
        if self.drifting:
            penalty_loss = self.drift(poi_output)
        else:
            penalty_loss = 0

        return poi_output, penalty_loss

