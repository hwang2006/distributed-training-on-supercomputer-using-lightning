import os
import pandas as pd

from pprint import pprint

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR

#from pytorch_lightning import LightningModule, Trainer, seed_everything
from lightning import LightningModule, Trainer, seed_everything

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
from transformers import BertForSequenceClassification, BertTokenizer

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from lightning.pytorch.loggers import CSVLogger

import re
import emoji
from soynlp.normalizer import repeat_normalize

# Download datasets
if not os.path.exists('./nsmc'):
    os.system("git clone https://github.com/e9t/nsmc")

args = {
    "random_seed": 42, # Random Seed
    "pretrained_model": "beomi/KcELECTRA-base-v2022",  # Transformers PLM name
    #"pretrained_model": "beomi/kcbert-base", 
    #"pretrained_model": "beomi/kcbert-large", 
    "pretrained_tokenizer": "",  # Optional, Transformers Tokenizer Name. Overrides `pretrained_model`
    "batch_size": 32,
    "lr": 5e-6,  # Starting Learning Rate
    "epochs": 1,  # Max Epochs
    "max_length": 150,  # Max Length input size
    "report_cycle": 100,  # Report (Train Metrics) Cycle
    "train_data_path": "nsmc/ratings_train.txt",  # Train Dataset file 
    "val_data_path": "nsmc/ratings_test.txt",  # Validation Dataset file 
    "test_mode": False,  # Test Mode enables `fast_dev_run`
    "optimizer": "AdamW",  # AdamW vs AdamP
    "lr_scheduler": "exp",  # ExponentialLR vs CosineAnnealingWarmRestarts
    #"fp16": True,  # Enable train on FP16(if GPU)
    "fp16": False,
    "tpu_cores": 0,  # Enable TPU with 1 core or 8 cores
    "cpu_workers": os.cpu_count(),
}

class Model(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters() #storing **kwargs to self.hparams
        self.validation_step_outputs = []
        self.train_step_outputs = []
        
        self.clsfier = AutoModelForSequenceClassification.from_pretrained(self.hparams.pretrained_model)
        #self.clsfier = BertForSequenceClassification.from_pretrained(self.hparams.pretrained_model)
        self.tokenizer = AutoTokenizer.from_pretrained(
        #self.tokenizer = BertTokenizer.from_pretrained(
            self.hparams.pretrained_tokenizer
            if self.hparams.pretrained_tokenizer
            else self.hparams.pretrained_model
        )

    def forward(self, **kwargs):
        return self.clsfier(**kwargs)

    def step(self, batch, batch_idx):
        data, labels = batch
        output = self(input_ids=data, labels=labels)

        # Transformers 4.0.0+
        loss = output.loss
        logits = output.logits

        preds = logits.argmax(dim=-1)

        y_true = list(labels.cpu().numpy())
        y_pred = list(preds.cpu().numpy())

        return {
            'loss': loss,
            'y_true': y_true,
            'y_pred': y_pred,
        }

    def training_step(self, batch, batch_idx):
        #return self.step(batch, batch_idx)
        #return self.train_step_outputs.append(self.step(batch, batch_idx)) #causing out of memory 
        step_outputs = self.step(batch, batch_idx)
        self.train_step_outputs.append(step_outputs)
        #print("I am in training_step")
        
        if (batch_idx % self.hparams.report_cycle) == 0:
           loss = step_outputs['loss']
           y_true = step_outputs['y_true']
           y_pred = step_outputs['y_pred']
           acc = accuracy_score(y_true, y_pred)
           prec = precision_score(y_true, y_pred)
           rec = recall_score(y_true, y_pred)
           f1 = f1_score(y_true, y_pred)
           print()
           print(f'[Epoch {self.trainer.current_epoch} Steps: {batch_idx}]:\n Loss: {loss}\n Acc: {acc}\n Prec: {prec}\n Rec: {rec}\n F1: {f1}\n')
           #print()
           #pprint(step_outputs)

        return step_outputs
        #return # returning nothing seems to lead to CUDA out of memory

    def validation_step(self, batch, batch_idx):
        #val_output = self.step(batch, batch_idx)
        #self.validation_step_outputs.append(val_output)
        #return self.step(batch, batch_idx)
        return self.validation_step_outputs.append(self.step(batch, batch_idx))

    def epoch_end(self, outputs, state='train'):
        loss=torch.tensor(0, dtype=torch.float)
        for i in outputs:
            loss += i['loss'].cpu().detach()
        loss = loss / len(outputs)

        y_true = []
        y_pred = []
        for i in outputs:
            y_true += i['y_true']
            y_pred += i['y_pred']
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        #self.log(state+'_loss', float(loss), on_epoch=True, prog_bar=True)
        #self.log(state+'_acc', acc, on_epoch=True, prog_bar=True)
        #self.log(state+'_precision', prec, on_epoch=True, prog_bar=True)
        #self.log(state+'_recall', rec, on_epoch=True, prog_bar=True)
        #self.log(state+'_f1', f1, on_epoch=True, prog_bar=True)
        print(f'[Epoch {self.trainer.current_epoch} {state.upper()}] Loss: {loss}, Acc: {acc}, Prec: {prec}, Rec: {rec}, F1: {f1}')
        #if state == "train":
        #    self.train_step_outputs.clear()
        #if state == "val":
        #    self.validation_step_outputs.clear()
        return {'loss': loss}
    
    #def training_epoch_end(self, outputs):
    #    self.epoch_end(outputs, state='train')
    def on_train_epoch_end(self):
        #print("I am in on_train_epoch_end")
        self.epoch_end(self.train_step_outputs, state='train')
        self.train_step_outputs.clear()
        #return        
        
    #def validation_epoch_end(self, outputs):
    #    self.epoch_end(outputs, state='val')
    def on_validation_epoch_end(self):
        self.epoch_end(self.validation_step_outputs, state='val')
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        if self.hparams.optimizer == 'AdamW':
            optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        elif self.hparams.optimizer == 'AdamP':
            from adamp import AdamP
            optimizer = AdamP(self.parameters(), lr=self.hparams.lr)
        else:
            raise NotImplementedError('Only AdamW and AdamP is Supported!')
        if self.hparams.lr_scheduler == 'cos':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)
        elif self.hparams.lr_scheduler == 'exp':
            scheduler = ExponentialLR(optimizer, gamma=0.5)
        else:
            raise NotImplementedError('Only cos and exp lr scheduler is Supported!')
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
        }
    
    def read_data(self, path):
        if path.endswith('xlsx'):
            return pd.read_excel(path)
        elif path.endswith('csv'):
            return pd.read_csv(path)
        elif path.endswith('tsv') or path.endswith('txt'):
            return pd.read_csv(path, sep='\t')
        else:
            raise NotImplementedError('Only Excel(xlsx)/Csv/Tsv(txt) are Supported')

    def preprocess_dataframe(self, df):
        emojis = ''.join(emoji.UNICODE_EMOJI.keys())
        pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
        url_pattern = re.compile(
            r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

        def clean(x):
            x = pattern.sub(' ', x)
            x = url_pattern.sub('', x)
            x = x.strip()
            x = repeat_normalize(x, num_repeats=2)
            return x

        df['document'] = df['document'].map(lambda x: self.tokenizer.encode(
            clean(str(x)),
            padding='max_length',
            max_length=self.hparams.max_length,
            truncation=True,
        ))
        return df

    def train_dataloader(self):
        df = self.read_data(self.hparams.train_data_path)
        df = self.preprocess_dataframe(df)

        dataset = TensorDataset(
            torch.tensor(df['document'].to_list(), dtype=torch.long),
            torch.tensor(df['label'].to_list(), dtype=torch.long),
        )
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size or self.batch_size,
            shuffle=True,
            num_workers=self.hparams.cpu_workers,
        )

    def val_dataloader(self):
        df = self.read_data(self.hparams.val_data_path)
        df = self.preprocess_dataframe(df)

        dataset = TensorDataset(
            torch.tensor(df['document'].to_list(), dtype=torch.long),
            torch.tensor(df['label'].to_list(), dtype=torch.long),
        )
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size or self.batch_size,
            shuffle=False,
            num_workers=self.hparams.cpu_workers,
        )

from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    filename='epoch{epoch}-val_acc{val_acc:.4f}',
    monitor='val_acc',
    save_top_k=3,
    mode='max',
    auto_insert_metric_name=False,
)

args["cpu_workers"] = 4
args["batch_size"] = 64
#args["batch_size"] = 256
args["fp16"] = True
args["epochs"] = 5
#os.environ['TOKENIZERS_PARALLELISM'] = "False"
os.environ['TOKENIZERS_PARALLELISM'] = "True"


def main(cliargs):
	print("Using PyTorch Ver", torch.__version__)
	print("Fix Seed:", args['random_seed'])
	print("cliargs: ", cliargs)
	#print(os.system("env | grep SLURM"))
	seed_everything(args['random_seed'])

	#args["cpu_workers"] = 4
	#args["batch_size"] = 128
	#args["batch_size"] = 256
	#args["fp16"] = True
	#args["epochs"] = 5
	#os.environ['TOKENIZERS_PARALLELISM'] = "False"
	#os.environ['TOKENIZERS_PARALLELISM'] = "True"

	model = Model(**args)

	print(":: Start Training ::")
	trainer = Trainer(
		#callbacks=[checkpoint_callback],
		max_epochs=args['epochs'],
		fast_dev_run=args['test_mode'],
		num_sanity_val_steps=None if args['test_mode'] else 0,
		# For GPU Setup
		#deterministic=torch.cuda.is_available(),
		#gpus=[0] if torch.cuda.is_available() else None,  # 0번 idx GPU  사용
		accelerator=cliargs.accelerator,
		strategy=cliargs.strategy,
		#devices=torch.cuda.device_count() if torch.cuda.is_available() else None,
		devices=cliargs.devices,
		num_nodes = cliargs.num_nodes,
		precision=16 if args['fp16'] else 32,
		# For TPU Setup
		# tpu_cores=args['tpu_cores'] if args['tpu_cores'] else None,
		logger=CSVLogger(save_dir="logs1/")
	)
	trainer.fit(model)

from argparse import ArgumentParser

if __name__ ==  '__main__':
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default="gpu" if torch.cuda.is_available() else "auto")
    #parser.add_argument("--accelerator", default="gpu" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--devices", default=torch.cuda.device_count() if torch.cuda.is_available() else None)
    #parser.add_argument("--strategy", default="ddp" if torch.cuda.is_available() else "auto")
    parser.add_argument("--strategy", default="ddp" if torch.cuda.is_available() else None)
    parser.add_argument("--num_nodes", default=1)
    cliargs = parser.parse_args()

    if os.getenv('SLURM_NTASKS_PER_NODE') is not None:
       cliargs.devices = os.getenv('SLURM_NTASKS_PER_NODE') # devices to be set to the slurm argument

    main(cliargs)
