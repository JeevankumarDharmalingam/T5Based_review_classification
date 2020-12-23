import argparse
args_dict = dict(
    output_dir = "Output_Directory",
    data_dir = "all-data.csv",
    model_name = "t5-base",
    tokenizer_name = "t5-base",
    max_seq_length = 512,
    lr = 0.0001,
    weight_decay = 0.0,
    adam_epsilon = 1e-8,
    warmup_steps = 0,
    train_batch_size = 1,
    eval_batch_size = 1,
    num_train_epochs = 2,
    gradient_accumulation_steps = 8,
    n_gpu = 1,
    seed = 42,
    early_stop_callback = False,
    fp_16 = False,
)


args = argparse.Namespace(**args_dict)