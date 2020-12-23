import pandas as pd
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import Dataset, DataLoader
from Data_Prep import Review_Dataset
from Engine import Review_Engine
from Utils import args
from transformers import T5Tokenizer


if __name__ == '__main__':
    df = pd.read_csv('all-data.csv', encoding="ISO-8859-1", names=['emotion', 'text'])
    X = df['text']
    Y = df['emotion']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.27, shuffle=True)
    X_train = X_train.reset_index(drop=True)

    X_test = X_test.reset_index(drop=True)

    Y_train = Y_train.reset_index(drop=True)

    Y_test = Y_test.reset_index(drop=True)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=args.output_dir,prefix="checkpoint",monitor="avg_loss",mode="min",verbose=1,save_top_k=5
    )
    model = Review_Engine(hparams=args,Train_Source=X_train,Train_target=Y_train,Val_Source=X_test,Val_Target=Y_test)
    trainer = Trainer(gpus=1,max_epochs=args.num_train_epochs,progress_bar_refresh_rate=10,
                      accumulate_grad_batches=args.gradient_accumulation_steps,checkpoint_callback=checkpoint_callback,
                      early_stop_callback=False)
    trainer.fit(model)