from tools.parser import parser
from tools.utils import load_config
import wandb
from trainer import Trainer
from tools.render_prediction import render_prediction


def main(config):

    trainer = Trainer(config)
    if config['wandb']['use']:
        wandb_args = {'project': trainer.config['wandb']['project'], 'name': trainer.config['wandb']['name'],
                      'config': trainer.config, 'tags': trainer.config['wandb']['tags'], 'group': trainer.config['wandb']['group']}
        wandb.init(**wandb_args)
        if config['wandb']['log_gradients']:
            wandb.watch(trainer.model.encoder, idx=0, log='all')
            wandb.watch(trainer.model.decoder, idx=1, log='all')
    if 'train' in config['arguments']['task']:
        trainer.train()
    if 'test' in config['arguments']['task']:
        trainer.test()
        if 'render' in config['arguments']['task']:
            save_images = trainer.save_path_results / f"test_{trainer.ckpt['epoch']}_images"
            save_images.mkdir()
            render_prediction(trainer, save_images).render()
    wandb.finish()


if __name__ == '__main__':
    arguments = parser()
    config = load_config(arguments.config, arguments)
    main(config)