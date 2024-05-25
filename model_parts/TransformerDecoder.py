import torch
from model_parts.utils import TokenEmbedding, PositionalEncoding, create_mask


class DecoderTransformer(torch.nn.Module):
    def __init__(self, num_decoder_layers: int,
                 emb_size: int, tgt_vocab_size: int,
                 dim_feedforward: int = 512, n_head=8, dropout: float = 0.1, softmax=False, vocab=None, max_len=150):
        super(DecoderTransformer, self).__init__()

        decoder_layer = torch.nn.TransformerDecoderLayer(d_model=emb_size, nhead=n_head,
                                                dim_feedforward=dim_feedforward)
        self.transformer_decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        if softmax:
            self.classifier = torch.nn.Sequential(torch.nn.Linear(emb_size, tgt_vocab_size), torch.nn.LogSoftmax(dim=-1))
        else:
            self.classifier = torch.nn.Sequential(torch.nn.Linear(emb_size, tgt_vocab_size))
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
        self.train_generator = False
        self.vocab = vocab
        self.max_len = max_len

    def _forward(self, features, targets, target_mask=None, target_padding_mask=None):
        if self.train_generator:
            with torch.no_grad():
                tgt_emb = self.positional_encoding(self.tgt_tok_emb(targets))
                outs = self.transformer_decoder(tgt_emb, features, target_mask, None, target_padding_mask)
        else:
            tgt_emb = self.positional_encoding(self.tgt_tok_emb(targets))
            outs = self.transformer_decoder(tgt_emb, features, target_mask, None, target_padding_mask)  # (tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None)#
        # tgt – the sequence to the decoder (required).

        # memory – the sequence from the last layer of the encoder (required).

        # tgt_mask – the mask for the tgt sequence (optional).

        # memory_mask – the mask for the memory sequence (optional).

        # tgt_key_padding_mask – the mask for the tgt keys per batch (optional).

        # memory_key_padding_mask – the mask for the memory keys per batch (optional).
        return self.classifier(outs)

    def forward(self, features, labels):
        labels_tgt = labels.permute(1, 0)
        if self.training:
            tgt_inputs = labels_tgt[:-1, :]
            tgt_masks, tgt_padding_masks = create_mask(tgt_inputs, labels.device, self.vocab.token2id['_PAD_'])
            out = self._forward(features, tgt_inputs, tgt_masks, tgt_padding_masks)
        else:
            # out = torch.ones((self.max_len, len(labels), self.vocab.__len__()))
            out = []
            for labels_i in range(len(labels)):
                batch_outputs = []
                tgt_inputs = torch.ones(1, 1) * self.vocab.token2id['_START_']
                tgt_inputs = tgt_inputs.to(torch.long).to(labels.device)
                while True:
                    tgt_masks, tgt_padding_masks = create_mask(tgt_inputs, labels.device, self.vocab.token2id['_PAD_'])
                    outputs = self._forward(features[:, labels_i:labels_i + 1, :], tgt_inputs, tgt_masks,
                                           tgt_padding_masks)
                    pred = torch.argmax(outputs, -1)
                    tgt_inputs = torch.cat([tgt_inputs, pred[-1:, :]], 0)
                    pred = pred[-1].item()
                    batch_outputs.append(outputs)
                    if pred == self.vocab.token2id['_END_'] or tgt_inputs.size(0) >= self.max_len:
                        break
                out.append(outputs)
        return out