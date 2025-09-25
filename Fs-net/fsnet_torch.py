# coding=utf-8
import torch
import torch.nn as nn


class FsNetTorch(nn.Module):
    """
    PyTorch implementation of the TensorFlow Fs_net.tinny_fs_net model in `Fs-net/model.py`.
    - Encoder: 2-layer bidirectional GRU, hidden size = 128
    - Decoder: 2-layer bidirectional GRU, input is the tiled encoder features across time
      hidden size = 128 (to match TF where `decoder_n_neurons = vocab_size = 128`)
    - Classification head: SELU-activated hidden layer (128), then Linear to 4 classes
    """

    def __init__(
        self,
        n_steps: int = 256,
        n_inputs: int = 1,
        n_outputs: int = 4,
        n_neurons: int = 128,
        encoder_n_neurons: int = 128,
        decoder_n_neurons: int = 128,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_neurons = n_neurons
        self.encoder_n_neurons = encoder_n_neurons
        self.decoder_n_neurons = decoder_n_neurons

        # Encoder: 2-layer BiGRU
        self.encoder = nn.GRU(
            input_size=n_inputs,
            hidden_size=encoder_n_neurons,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        # Decoder: input is encoder_feats tiled across time (dim = 2*encoder_hidden)
        self.decoder = nn.GRU(
            input_size=2 * encoder_n_neurons,
            hidden_size=decoder_n_neurons,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        # Classifier input features: concat of (encoder, decoder, element-wise product, abs diff)
        # encoder_feats: (batch, 2*encoder_hidden)
        # decoder_feats: (batch, 2*decoder_hidden)
        # TF sets encoder_hidden=128 and decoder_hidden=128, so dims match (=256)
        cls_in_dim = 4 * (2 * encoder_n_neurons)  # 4 * 256 = 1024
        self.fc1 = nn.Linear(cls_in_dim, n_neurons)
        self.act1 = nn.SELU()
        self.fc2 = nn.Linear(n_neurons, n_outputs)
        # TF tinny_fs_net applies SELU on the second dense as well (non-standard for logits)
        # We mirror that here for functional parity
        self.act2 = nn.SELU()

        self._init_weights()

    @staticmethod
    def _last_layer_bi_hidden(h: torch.Tensor) -> torch.Tensor:
        """
        h shape: (num_layers * num_directions, batch, hidden_size)
        For 2 layers bidirectional -> order: [l0_f, l0_b, l1_f, l1_b]
        Take the last layer's forward/backward hidden and concat => (batch, 2*hidden)
        """
        h_fwd_last = h[-2]  # (batch, hidden)
        h_bwd_last = h[-1]  # (batch, hidden)
        return torch.cat([h_fwd_last, h_bwd_last], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_steps, n_inputs)
        enc_out, enc_h = self.encoder(x)  # enc_h: (4, batch, hidden)
        encoder_feats = self._last_layer_bi_hidden(enc_h)  # (batch, 2*enc_hidden)

        # Tile encoder feats across time to feed decoder
        dec_in = encoder_feats.unsqueeze(1).repeat(1, self.n_steps, 1)  # (batch, n_steps, 2*enc_hidden)
        dec_out, dec_h = self.decoder(dec_in)
        decoder_feats = self._last_layer_bi_hidden(dec_h)  # (batch, 2*dec_hidden)

        # Element-wise interactions (dims match because enc_hidden == dec_hidden == 128)
        element_wise_product = encoder_feats * decoder_feats
        element_wise_absolute = torch.abs(encoder_feats - decoder_feats)

        cls_feats = torch.cat([
            encoder_feats,
            decoder_feats,
            element_wise_product,
            element_wise_absolute,
        ], dim=-1)  # (batch, 1024)

        x = self.fc1(cls_feats)
        x = self.act1(x)
        logits = self.fc2(x)
        logits = self.act2(logits)
        return logits

    def _init_weights(self):
        # Initialize GRUs: Xavier for input weights, Orthogonal for recurrent weights; zero biases
        for name, param in self.encoder.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        for name, param in self.decoder.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        # Linear layers: Xavier for weights, zeros for biases
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
