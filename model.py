import torch
from vit_pytorch import ViT
from model_parts.cvt import CvT
from tools.utils import get_accuracy
from model_parts.regionvit import RegionViT
from model_parts.swin_transformer_v2 import SwinTransformerV2
from model_parts.CNNEncoder import CNNEncoder
from model_parts.TransformerDecoder import DecoderTransformer


class model(torch.nn.Module):
    def __init__(self, config, vocab=None, styles=None):
        super(model, self).__init__()
        self.config = config
        self.styles = styles
        if vocab:
            self.vocab = vocab
        if self.config['model']['encoder'] == 'cnn':
            self.config['model']['cnn']['cnn_features'] = self.config['model']['encoder_emb']
            self.encoder = CNNEncoder(self.config['model']['cnn']).to(self.config['device'])
        elif self.config['model']['encoder'] == 'vit':
            self.encoder = ViT(image_size=self.config['model']['image']['width'],
                               patch_size=self.config['model']['vit']['patch_size'],
                               num_classes=self.config['model']['encoder_emb'],
                               dim=self.config['model']['vit']['dim'], depth=self.config['model']['vit']['depth'],
                               heads=self.config['model']['vit']['heads'],
                               mlp_dim=self.config['model']['vit']['mlp_dim'],
                               dropout=self.config['model']['vit']['dropout'],
                               emb_dropout=self.config['model']['vit']['emb_dropout'],
                               channels=self.config['model']['vit']['channels']).to(self.config['device'])
        elif self.config['model']['encoder'] == 'CvT':
            self.encoder = CvT(s3_emb_dim=self.config['model']['encoder_emb'],  #384,   stage 3 - (same as above)
                               **self.config['model']['cvt']).to(self.config['device'])
        elif self.config['model']['encoder'] == 'RegionViT':
            self.encoder = RegionViT(
                dim=(64, 128, 256, 512),  # tuple of size 4, indicating dimension at each stage
                depth=(2, 2, 8, 2),  # depth of the region to local transformer at each stage
                window_size=14,  # window size, which should be either 7 or 14
                num_classes=self.config['model']['encoder_emb'],  # number of output classes
                tokenize_local_3_conv=False,
                # whether to use a 3 layer convolution to encode the local tokens from the image. the paper uses this for the smaller models, but uses only 1 conv (set to False) for the larger models
                use_peg=False,
                # whether to use positional generating module. they used this for object detection for a boost in performance
                local_patch_size=8,
                channels=1
            ).to(self.config['device'])
        elif self.config['model']['encoder'] == 'swin':
            self.encoder = SwinTransformerV2(
                img_size=(self.config['model']['image']['height'], self.config['model']['image']['width']),
                embed_dim=int(self.config['model']['encoder_emb']/(2**(len(self.config['model']['swin']['depths'])-1)))
                , **self.config['model']['swin']
            ).to(self.config['device'])
        else:
            raise Exception(f"encoder {self.config['model']['encoder']} not available, please choose one of [cnn, vit, wav2vec2]")
        if self.config['model']['decoder'] == 'transformer':
            self.decoder = DecoderTransformer(num_decoder_layers=self.config['model']['transformer']['num_decoder_layers'],
                                              emb_size=self.config['model']['encoder_emb'],
                                              tgt_vocab_size=self.config['model']['vocab_size'],
                                              dim_feedforward=self.config['model']['transformer']['hidden_size'],
                                              n_head=self.config['model']['transformer']['n_head'],
                                              dropout=self.config['model']['transformer']['dropout'],
                                              softmax=self.config['model']['softmax'],
                                              vocab=self.vocab, max_len=self.config['model']['max_len']).to(
                self.config['device'])
        elif self.config['model']['decoder'] == 'linear':
            self.decoder = torch.nn.Sequential(torch.nn.Linear(self.config['model']['encoder_emb'], self.vocab.__len__()).to(self.config['device']), torch.nn.LogSoftmax(dim=-1))
            self.decoder.train_generator = False
        else:
            raise Exception(f"decoder {self.config['model']['decoder']} not available, please choose one of [transformer, linear]")
        if self.config['train']['criterion'] == 'CrossEntropyLoss':
            self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        if self.config['train']['predict_style']:
            self.style_classifier = torch.nn.Linear(self.config['model']['encoder_emb'], len(styles)).to(self.config['device'])
            self.style_softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, images, labels, epoch):
        # encoder
        self.encoder.requires_grad_(True)
        if epoch <= self.config['train']['freeze_feature_extractor']:
            self.encoder._modules['feature_extractor'].requires_grad_(False)
        if epoch <= self.config['train']['freeze_encoder']:
            self.encoder.requires_grad_(False)
        if self.config['model']['encoder'] == 'wav2vec2':
            features = self.encoder(images, mask=False, features_only=True)['x'].to(self.config['device'])
            features = features.permute(1, 0, 2)
        elif self.config['model']['encoder'] == 'vit' or self.config['model']['encoder'] == 'swin':
            features = self.encoder(images).to(self.config['device'])
            features = torch.unsqueeze(features, dim=0)
        else:
            features = self.encoder(images).to(self.config['device'])  # (t x b x c)
        # decoder
        labels_tgt = labels.permute(1, 0)
        if self.config['model']['decoder'] == 'transformer':
            out = self.decoder(features, labels)
            if self.decoder.training:
                total_loss = self.criterion(out.permute((0, 2, 1)), labels_tgt[1:])
                predictions = torch.argmax(out.permute(1, 0, 2), -1)
            else:
                total_loss = 0
                predictions = []
                for out_i in out:
                    predictions.append(torch.argmax(out_i, -1).squeeze(dim=1))
        else:
            out = self.decoder(features)
            total_loss = self.criterion(out.permute((1, 2, 0)), labels_tgt[1:])
            predictions = torch.argmax(out, -1)
        return predictions, total_loss, get_accuracy(labels=labels, predictions=predictions, end_number=self.vocab.token2id['_END_'])

    def forward_self_supervised(self, images, labels, epoch):
        # first and second round
        # encoder
        self.encoder.requires_grad_(True)
        if epoch <= self.config['train']['freeze_feature_extractor']:
            self.encoder._modules['feature_extractor'].requires_grad_(False)
        if epoch <= self.config['train']['freeze_encoder']:
            self.encoder.requires_grad_(False)
        if self.config['model']['encoder'] == 'wav2vec2':
            features = self.encoder(images, mask=False, features_only=True)['x'].to(self.config['device'])
            features = features.permute(1, 0, 2)
        elif self.config['model']['encoder'] == 'vit' or self.config['model']['encoder'] == 'swin':
            features = self.encoder(images).to(self.config['device'])
            features = torch.unsqueeze(features, dim=0)
        else:
            features = self.encoder(images).to(self.config['device'])  # (t x b x c)
        # decoder
        labels_tgt = labels.permute(1, 0)
        if self.config['model']['decoder'] == 'transformer':
            out = self.decoder(features, labels)
            if self.decoder.training:
                total_loss = self.criterion(out.permute((0, 2, 1)), labels_tgt[1:])
                predictions = torch.argmax(out.permute(1, 0, 2), -1)
            else:
                total_loss = 0
                predictions = []
                for out_i in out:
                    predictions.append(torch.argmax(out_i, -1).squeeze(dim=1))
        else:
            out = self.decoder(features)
            total_loss = self.criterion(out.permute((1, 2, 0)), labels_tgt[1:])
            predictions = torch.argmax(out, -1)
        return predictions, total_loss, get_accuracy(labels=labels, predictions=predictions, end_number=self.vocab.token2id['_END_'])