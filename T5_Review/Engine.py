

import torch
import torch.nn as nn
import pytorch_lightning as pl
from cachetools.func import lru_cache
from transformers import T5ForConditionalGeneration,T5Tokenizer,AdamW,get_linear_schedule_with_warmup
from Data_Prep import Review_Dataset
from Utils import args
from torch.utils.data import Dataset, DataLoader
import numpy as np

class Review_Engine(pl.LightningModule):
    def __init__(self,hparams,Train_Source,Train_target,Val_Source,Val_Target):
        super(Review_Engine, self).__init__()
        self.hparams = hparams
        self.model = T5ForConditionalGeneration.from_pretrained(self.hparams.model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.tokenizer_name)
        self.Train_Source = Train_Source
        self.Train_target = Train_target
        self.Val_Source = Val_Source
        self.Val_Target = Val_Target

    def forward(self, inputs_ids,attention_mask = None,decoder_input_ids=None,decoder_attention_mask = None,
                lm_labels = None):
        return self.model(inputs_ids,attention_mask=attention_mask,decoder_input_ids=decoder_input_ids,
                          decoder_attention_mask = decoder_attention_mask,lm_labels = lm_labels)

    def _step(self,batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:,:] == self.tokenizer.pad_token_id] = -100

        outputs = self(inputs_ids = batch["source_ids"],
                       attention_mask = batch["source_mask"],
                       lm_labels = lm_labels,
                       decoder_attention_mask = batch["target_mask"])

        loss = outputs[0]
        return loss

    def training_step(self,batch,batch_idx):
        loss = self._step(batch)
        tensorboard_logs = {"train_loss":loss}

        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self,batch,batch_idx):
        loss = self._step(batch)

        return {"val_loss": loss}

    def training_epoch_end(self,outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}

        return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, "progress_bar": tensorboard_logs}

    def validation_epoch_end(self,outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_loss": avg_loss}

        return {"avg_loss": avg_loss, "log": tensorboard_logs, "progress_bar": tensorboard_logs}

    # def configure_optimizers(self):
    #
    #
    #     optimizer = AdamW(model.parameters(),lr=self.hparams.lr)
    #     lr_scheduler = get_linear_schedule_with_warmup(
    #         optimizer=optimizer,
    #         num_warmup_steps=self.hparams.warmup_steps,
    #         num_training_steps= self.total_steps()
    #     )
    #
    #     return [optimizer],[{"scheduler":lr_scheduler,"interval":"step"}]

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.lr, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None, on_tpu=False,
                       using_native_amp=False, using_lbfgs=False):
        if self.trainer.use_tpu:
            #xm.optimizer_step(optimizer)
            print()
        else:
            optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

        return tqdm_dict

    def train_dataloader(self):
        dataset = Review_Dataset(self.Train_Source,self.Train_target,self.tokenizer)
        dataloader = DataLoader(dataset,batch_size=self.hparams.train_batch_size,shuffle=True,num_workers=4)
        t_total = (
                (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = Review_Dataset(self.Val_Source, self.Val_Target, self.tokenizer)
        val_dataloader = DataLoader(val_dataset,batch_size = self.hparams.eval_batch_size,num_workers=4)
        return val_dataloader


