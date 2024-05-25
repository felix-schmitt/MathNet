import pathlib
import numpy as np
import torch
import yaml

import model
from tqdm import tqdm
from tools.utils import score_files
import wandb
import datetime
import shutil
from tools.dataloader import Im2LatexDataset


class Trainer():
    def __init__(self, config):
        # save config
        self.config = config
        # prepare save folder
        e = datetime.datetime.now()
        self.save_path = pathlib.Path(
            self.config['model']['model_save_path']) / f'run_{e.year}-{e.month}-{e.day}_{e.hour}-{e.minute}-{e.second}'
        self.config['model']['model_save_path'] = str(self.save_path)
        self.config['wandb']['name'] += f"-{e.year}-{e.month}-{e.day}_{e.hour}-{e.minute}-{e.second}"
        print("save path")
        pathlib.Path(self.save_path).mkdir(exist_ok=True, parents=True)
        self.save_path_results = self.save_path / 'results'
        self.save_path_results.mkdir(exist_ok=True, parents=True)
        self.save_path_model = self.save_path / 'model'
        self.save_path_model.mkdir(exist_ok=True, parents=True)

        # prepare dataloader
        print("load test dataset")
        # load test dataset
        self.test_dataloader = {name: self._load_dataset(config_files, mode='test') for
                                name, config_files in self.config['dataset']['test']['files'].items()}
        for name, dataloader in self.test_dataloader.items():
            self._save_gt(f'test_{name}', dataloader)

        # load train and val datasets if needed
        if 'train' in self.config['arguments']['task']:
            print("load val dataset")
            self.val_dataloader = self._load_dataset(self.config['dataset']['val']['file'], mode='val')
            self._save_gt('val', self.val_dataloader)

            print("load train dataset")
            self.train_dataloader = self._load_dataset(self.config['dataset']['train']['file'], mode='train')

            self.lazy_cache_train = self.config['dataset']['train']['cache']
        self.vocab = self.test_dataloader[list(self.test_dataloader.keys())[0]].dataset.vocab

        print("Create model")
        self.model = model.model(self.config, self.vocab, self.train_dataloader.dataset.styles if 'train' in self.config['arguments']['task'] else None)
        self.optimizer = torch.optim.Adam(self.model.parameters(), config['train']['init_lr'])

        self.ckpt = {'epoch': 0,
                     'best_fitness': 0,
                     'encoder_model': None,
                     'decoder_model': None,
                     'optimizer': None,
                     'train_loss': [],
                     'train_accuracy': [],
                     'val_loss': [],
                     'val_accuracy': []
                     }
        if self.config['arguments']['resume_from']:
            print("loading model")
            self._load_model()
        self.epoch = self.ckpt['epoch']

        # count parameters
        params = 0
        for p in self.model.decoder.parameters():
            params += np.prod(p.size())
        for p in self.model.encoder.parameters():
            params += np.prod(p.size())
        print("total parameters = ", params)

        # save vocab and config
        shutil.copyfile(self.config['dataset']['vocab_file'], self.save_path / "latex.vocab")
        with open(self.save_path / 'config.yml', 'w') as outfile:
            yaml.dump(self.config, outfile, default_flow_style=False)

    def _save_gt(self, mode, dataloader):
        lazy_cache = self.config['dataset'][mode.split("_")[0]]['cache']
        with open(self.save_path/f'gt_{mode}.txt', "w") as file:
            for images, labels in tqdm(dataloader, postfix=f"save {mode} dataset"):
                formulas = labels[1].cpu().numpy()
                for label_i, label in enumerate(labels[0][0]):
                    text = label + ": " + dataloader.dataset.vocab.seq2text(formulas[label_i]) + '\n'
                    file.write(text)
            if lazy_cache:
                dataloader.dataset.cache_all_files()
                dataloader.num_workers = self.config['dataset'][mode.split("_")[0]]['num_workers']

    def _load_dataset(self, dataset_config_files, mode):
        dataset = Im2LatexDataset(dataset_config_files, self.config['dataset']['vocab_file'],
                                  self.config['model']['image'], mode=mode, cache=self.config['dataset'][mode]['cache'],
                                  max_size=self.config['model']['max_len'],
                                  transforms=self.config['dataset'][mode]['transforms'],
                                  no_sampling=self.config['dataset'][mode]['no_sampling'],
                                  no_arrays=self.config['dataset'][mode]['no_arrays'],
                                  only_basic=self.config['dataset'][mode]['only_basic'],
                                  normalize=self.config['dataset'][mode]['normalize'],
                                  dpi=self.config['dataset']['dpi'])
        ds_loader = torch.utils.data.DataLoader(dataset, batch_size=self.config['dataset'][mode]['batch_size'],
                                                shuffle=self.config['dataset'][mode]['shuffle'],
                                                drop_last=self.config['dataset'][mode]['drop_last'],
                                                num_workers=0 if self.config['dataset'][mode]['cache'] else self.config['dataset'][mode]['num_workers'])
        self.config['model']['vocab_size'] = len(dataset.vocab.id2token)
        return ds_loader

    def _load_model(self):
        try:
            self.ckpt = torch.load(self.config['arguments']['resume_from'])
            self.model.decoder.load_state_dict(self.ckpt['decoder_model'], strict=True)
            self.model.encoder.load_state_dict(self.ckpt['encoder_model'], strict=True)
            self.optimizer.load_state_dict(self.ckpt['optimizer'])
            print("load model from checkpoint")
        except Exception as e:
            print('Error occurred during loading ', self.config['arguments']['resume_from'], ' (', e, ')')

    def _train_batch(self, labels, images):
        labels = labels[1]
        self.model.encoder.zero_grad()
        self.model.decoder.zero_grad()
        images = images.to(torch.float).to(self.config['device'])
        labels = labels.to(self.config['device'])
        if self.config['train']['fp16']:
            with torch.amp.autocast(device_type='cuda' if 'cuda' in self.config['device'] else 'cpu'):
                outputs, loss, acc = self.model(images, labels, self.epoch)
        else:
            outputs, loss, acc = self.model(images, labels, self.epoch)
        word_loss = loss.item()
        self.optimizer.zero_grad()
        if self.config['train']['fp16']:
            self.scaler.scale(loss).backward()
            if self.config['train']['grad_clip_value']:
                torch.nn.utils.clip_grad_value_(self.model.encoder.parameters(), self.config['train']['grad_clip_value'])
                torch.nn.utils.clip_grad_value_(self.model.decoder.parameters(), self.config['train']['grad_clip_value'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.config['train']['grad_clip_value']:
                torch.nn.utils.clip_grad_value_(self.model.encoder.parameters(), self.config['train']['grad_clip_value'])
                torch.nn.utils.clip_grad_value_(self.model.decoder.parameters(), self.config['train']['grad_clip_value'])
            self.optimizer.step()
        loss_value = loss.item()
        return loss_value, acc, word_loss

    def train(self):
        pationt = 0
        best_fitness = 0
        if self.config['train']['fp16']:
            self.scaler = torch.cuda.amp.GradScaler()
        train_image = None
        for epoch in range(self.ckpt['epoch'], self.config['train']['epochs']):
            self.epoch = epoch
            # train
            self.model.encoder.train()
            self.model.decoder.train()
            mean_loss = 0
            mean_accuracy = 0
            with tqdm(self.train_dataloader) as pbar:
                for itr, (images, labels) in enumerate(pbar):
                    loss_value, acc, word_loss = self._train_batch(labels, images)
                    mean_loss += 1 / (itr + 1) * (loss_value - mean_loss)
                    mean_accuracy += 1 / (itr + 1) * (acc.item() - mean_accuracy)
                    s = 'EPOCH[%d/%d] loss=%2.4f - accuracy=%2.4f' % (
                        self.epoch, self.config['train']['epochs'], mean_loss, mean_accuracy)
                    pbar.set_description(s)
                    if self.config['wandb']['train_image'] and self.config['wandb']['train_image'] in labels[0][0]:
                        train_image = wandb.Image(images[labels[0][0].index(self.config['wandb']['train_image'])])
                    if itr % 100 == 0 and self.config['wandb']['use']:
                        if train_image:
                            wandb.log({"train/image": train_image})
                            train_image = None
                        wandb.log({"train/loss": mean_loss, "train/accuracy": mean_accuracy,
                                   "epoch": pbar.format_dict['n'] / pbar.format_dict['total'] + epoch,
                                   "train/word_loss": word_loss})
                if self.lazy_cache_train:
                    self.train_dataloader.dataset.cache_all_files()
                    self.train_dataloader.num_workers = self.config['dataset']['train']['num_workers']
                    self.lazy_cache_train = False
            self.ckpt['train_loss'].append(mean_loss)
            self.ckpt['train_accuracy'].append(mean_accuracy)

            if (self.epoch + 1) % self.config['train']['lr_descent'][0] == 0:
                if self.config['wandb']['use']:
                    wandb.log({"epoch": self.epoch, "train/lr": self.optimizer.param_groups[0]['lr']})
                for g in self.optimizer.param_groups:
                    g['lr'] = self.config['train']['lr_descent'][1] * g['lr']

            if self.epoch % self.config['train']['save_each'] == 0:
                self._save_model(f'epoch_{self.epoch}.pt')

            # validation
            if self.epoch % self.config['val']['each'] == 0 and self.epoch > self.config['train']['wait_n_epochs']:
                metrics, wandb_table = self.validate()
                self._wandb_log(metrics, wandb_table, 'val')
                # check if early stop or best
                if pationt > self.config['train']['early_stop']:
                    print("early stopping ... ")
                    break
                if metrics[self.config['val']['metric']] > best_fitness:
                    pationt = 0
                    best_fitness = metrics[self.config['val']['metric']]
                    self.ckpt['best_fitness'] = best_fitness
                    print("save best at accuracy = ", best_fitness)
                    self._save_model('best.pt')
                else:
                    pationt += 1

            # testing
            if epoch % self.config['test']['each'] == 0 and self.epoch > self.config['train']['wait_n_epochs']:
                metrics, wandb_table = self.test()
                for name, metric in metrics.items():
                    self._wandb_log(metric, wandb_table[name], f'test-{name}')

    def _wandb_log(self, metrics, wandb_table, mode):
        if not self.config['wandb']['use']:
            return
        wandb_metrics = {f"{mode}/" + key: val for key, val in metrics.items() if isinstance(val, float)}
        if self.config['wandb']['table']:
            wandb.log({"epoch": self.epoch, f"{mode}/example": wandb_table, **wandb_metrics})
        if self.more_information:
            data = [[key, value] for key, value in metrics['errors'].items()]
            table = wandb.Table(data=data[:self.config['wandb']['n_errors']], columns=["label", "value"])
            ops = {f"{mode}/Edit-" + key: val for key, val in metrics['ops'].items()}
            wandb.log({"epoch": self.epoch,
                       f"{mode}/token_errors": wandb.plot.bar(table, "label", "value", title="Custom Bar Chart"),
                       **ops, **wandb_metrics})
        else:
            wandb.log({"epoch": self.epoch, **wandb_metrics})

    def _evaluate(self, pred_path, dataloader, mode, more_information=False):
        self.more_information = more_information
        self.model.encoder.eval()
        self.model.decoder.eval()
        prediction_file = open(pred_path, 'w')
        # prepare one random batch for logging
        log_batch = np.random.randint(0, len(dataloader))
        max_images = 5
        table = []
        table_columns = ['image_id', 'image', 'gt', 'prediction']
        total_acc = []
        with torch.no_grad():
            with tqdm(dataloader, desc=f"{mode} ({self.epoch})") as data_loader:
                for batch_i, [images, labels] in enumerate(data_loader):
                    selection = []
                    image_names = labels[0][0]
                    if len(labels[0]) > 2:
                        style = labels[0][2]
                    else:
                        style = False
                    labels = labels[1]
                    images = images.to(torch.float).to(self.config['device'])  # better performance with fp16 .half().to(torch.float).to(self.config['device'])
                    labels = labels.to(self.config['device'])
                    outputs, loss, acc = self.model(images, labels, self.epoch)
                    total_acc.append(acc)
                    if batch_i == log_batch:
                        selection = np.array(range(len(images)))
                        np.random.shuffle(selection)
                        selection = selection[:min(len(images), max_images)]
                    for i in range(len(outputs)):
                        seq = list(outputs[i].cpu().numpy())
                        pred_text = self.vocab.seq2text(seq)
                        prediction_file.write(f"{image_names[i]}: " + pred_text)
                        if style:
                            prediction_file.write(f" style: {style[i]}")
                        prediction_file.write('\n')
                        if i in selection:
                            table.append([image_names[i], wandb.Image(images[i].cpu().numpy()),
                                          self.vocab.seq2text(labels[i].cpu().numpy()), pred_text])
        wandb_table = wandb.Table(data=table, columns=table_columns)
        prediction_file.close()
        gt = self.save_path / f"gt_{mode}.txt"
        metrics = score_files(gt, pred_path, more_information=self.more_information)
        metrics['acc'] = sum(total_acc) / len(total_acc)
        return metrics, wandb_table

    def validate(self):
        pred_path = self.save_path_results / f'val_{self.epoch}_predictions.txt'
        pred_path.parent.mkdir(exist_ok=True, parents=True)
        metrics, wandb_table = self._evaluate(pred_path, self.val_dataloader, f"val",
                                                        self.config['test']['more_information'])
        self._process_metrics(f'val_{self.epoch}_results.txt', metrics)
        return metrics, wandb_table

    def _save_model(self, file_name):
        # prepare checkpoint and save
        self.ckpt['epoch'] = self.epoch + 1
        self.ckpt['encoder_model'] = self.model.encoder.state_dict()
        self.ckpt['decoder_model'] = self.model.decoder.state_dict()
        self.ckpt['optimizer'] = self.optimizer.state_dict()
        torch.save(self.ckpt, self.save_path_model / file_name)

    def test(self):
        metrics = {}
        wandb_table = {}
        for name, dataloader in self.test_dataloader.items():
            pred_path = self.save_path_results / f'test_{self.epoch}_{name}_predictions.txt'
            pred_path.parent.mkdir(exist_ok=True, parents=True)
            temp_metrics, temp_wandb_table = self._evaluate(pred_path, dataloader, f"test_{name}", self.config['test']['more_information'])
            metrics[name] = temp_metrics
            wandb_table[name] = temp_wandb_table
            self._process_metrics(f'test_{self.epoch}_results.txt', metrics[name], name)
        return metrics, wandb_table

    def _process_metrics(self, filename, metrics, name=None):
        with open(str(self.save_path_results / filename), 'w') as f:
            if name:
                print(f"Dataset {name}:\n")
                f.write(f"Dataset {name}:\n")
            for key, value in metrics.items():
                if isinstance(value, dict):
                    f.write(f"\t{key}:\n")
                    for k, v in value.items():
                        f.write(f"\t\t{k}: {v}\n")
                else:
                    print(f"\t{key}: {value}")
                    f.write(f"\t{key}: {value}\n")
