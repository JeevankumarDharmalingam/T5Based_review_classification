from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import T5Tokenizer
from sklearn.model_selection import train_test_split


class Review_Dataset(Dataset):
    def __init__(self, text, emotion, tokenizer, max_len=512):
        self.text = text
        self.emotion = emotion
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.inputs_ = []
        self.targets = []

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        self.input_ = self.text[item] + "</s>"
        self.target = self.emotion[item] + "</s>"

        tokenized_inputs = self.tokenizer.batch_encode_plus([self.input_], max_length=self.max_len, padding='max_length',truncation=True,
                                                            return_tensors="pt")
        tokenized_target = self.tokenizer.batch_encode_plus([self.target], max_length=2,padding='max_length',truncation=True,
                                                            return_tensors="pt")

        return {
            "source_ids": tokenized_inputs["input_ids"].squeeze(),
            "source_mask": tokenized_inputs["attention_mask"].squeeze(),
            "target_ids": tokenized_target["input_ids"].squeeze(),
            "target_mask": tokenized_target["attention_mask"].squeeze()
        }


# if __name__ == '__main__':
#     df = pd.read_csv('all-data.csv', encoding="ISO-8859-1", names=['emotion', 'text'])
#     X = df['text']
#     Y = df['emotion']
#
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.27, shuffle=True)
#     X_train = X_train.reset_index(drop=True)
#     X_test = X_test.reset_index(drop=True)
#     Y_train = Y_train.reset_index(drop=True)
#     Y_test = Y_test.reset_index(drop=True)
#
#
#     tokenizer = T5Tokenizer.from_pretrained('t5-base')
#     dataset = Review_Dataset(X_train,Y_train,tokenizer=tokenizer)
#     data = dataset[42]
#     print(data['source_ids'])
#     print(tokenizer.decode(data['source_ids']))
#     print(tokenizer.decode(data['target_ids']))
#    print(len(dataset))





# tokenizer = T5Tokenizer.from_pretrained('t5-base')
# train_dataset = Review_Dataset(X_train, Y_train,tokenizer=tokenizer,max_len=512)
# valid_dataset = Review_Dataset(X_test, Y_test,tokenizer=tokenizer,max_len=512)
# data = train_dataset[0]
# data2 = train_dataset[1]
# print(data['source_ids'])
# print(data2['source_ids'])
# #     print(data['source_ids'])
# train_dataloader = DataLoader(train_dataset, batch_size=8, drop_last=True,shuffle=True,num_workers=8)
# valid_dataloader = DataLoader(valid_dataset, batch_size=8, num_workers=8)
#
# print(len(Y_train))