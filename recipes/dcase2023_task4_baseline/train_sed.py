import argparse
import os
import random
import warnings

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchaudio
import yaml
from desed_task.dataio import ConcatDatasetBatchSampler
from desed_task.dataio.datasets import (StronglyAnnotatedSet, UnlabeledSet,
                                        WeakSet)
from desed_task.nnet.CRNN import CRNN
from desed_task.utils.encoder import ManyHotEncoder
from desed_task.utils.schedulers import ExponentialWarmup
from local.classes_dict import classes_labels
from local.resample_folder import resample_folder
from local.sed_trainer import SEDTask4
from local.utils import calculate_macs, generate_tsv_wav_durations
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from functools import partial

feature_extraction = torch.nn.Sequential(
    torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        hop_length=320,
        n_mels=64,
    ),
    torchaudio.transforms.AmplitudeToDB(),
)


def resample_data_generate_durations(config_data, test_only=False, evaluation=False):
    if not test_only:
        dsets = [
            "synth_folder",
            "synth_val_folder",
            "strong_folder",
            "weak_folder",
            "unlabeled_folder",
            "test_folder",
        ]
    elif not evaluation:
        dsets = ["test_folder"]
    else:
        dsets = ["eval_folder"]

    for dset in dsets:
        computed = resample_folder(
            config_data[dset + "_44k"], config_data[dset], target_fs=config_data["fs"]
        )

    if not evaluation:
        for base_set in ["synth_val", "test"]:
            if not os.path.exists(config_data[base_set + "_dur"]) or computed:
                generate_tsv_wav_durations(
                    config_data[base_set + "_folder"], config_data[base_set + "_dur"]
                )


def single_run(
    config,
    log_dir,
    gpus,
    strong_real=False,
    checkpoint_resume=None,
    test_state_dict=None,
    fast_dev_run=False,
    evaluation=False,
    callbacks=None,
    use_filter_aug=True,
    filter_aug_prob=0.5,
    use_spec_aug=False,
    spec_aug_prob=0.5,
    use_time_stretch=True,
    time_stretch_prob=0.5,
):
    """
    Running sound event detection baseline

    Args:
        config (dict): the dictionary of configuration params
        log_dir (str): path to log directory
        gpus (int): number of gpus to use
        strong_real (bool): whether to use strong real annotations
        checkpoint_resume (str, optional): path to checkpoint to resume from
        test_state_dict (dict, optional): if not None, no training is involved
        fast_dev_run (bool, optional): whether to use a run with only one batch
        evaluation (bool): whether this is an evaluation run
        callbacks: pytorch lightning callbacks
        use_filter_aug (bool): whether to use filter augmentation
        filter_aug_prob (float): probability of applying filter augmentation
        use_spec_aug (bool): whether to use spec augmentation
        spec_aug_prob (float): probability of applying spec augmentation
        use_time_stretch (bool): whether to use time stretch augmentation
        time_stretch_prob (float): probability of applying time stretch
    """
    config.update({"log_dir": log_dir})

    # handle seed
    seed = config["training"]["seed"]
    if seed:
        pl.seed_everything(seed, workers=True)

    ##### data prep test ##########
    encoder = ManyHotEncoder(
        list(classes_labels.keys()),
        audio_len=config["data"]["audio_max_len"],
        frame_len=config["feats"]["n_filters"],
        frame_hop=config["feats"]["hop_length"],
        net_pooling=config["data"]["net_subsample"],
        fs=config["data"]["fs"],
    )

    if not evaluation:
        devtest_df = pd.read_csv(config["data"]["test_tsv"], sep="\t")
        devtest_dataset = StronglyAnnotatedSet(
            config["data"]["test_folder"],
            devtest_df,
            encoder,
            return_filename=True,
            pad_to=config["data"]["audio_max_len"],
            test=True,
        )
    else:
        devtest_dataset = UnlabeledSet(
            config["data"]["eval_folder"], 
            encoder, 
            pad_to=None, 
            return_filename=True
        )

    test_dataset = devtest_dataset

    ##### model definition  ############
    sed_student = CRNN(**config["net"])

    if test_state_dict is None:
        ##### data prep train valid ##########
        synth_df = pd.read_csv(config["data"]["synth_tsv"], sep="\t")
        synth_set = StronglyAnnotatedSet(
            config["data"]["synth_folder"],
            synth_df,
            encoder,
            pad_to=config["data"]["audio_max_len"],
            feats_pipeline=feature_extraction,
            use_filter_aug=use_filter_aug,
            filter_aug_prob=filter_aug_prob,
            use_spec_aug=use_spec_aug,
            spec_aug_prob=spec_aug_prob,
            use_time_stretch=use_time_stretch, 
            time_stretch_prob=time_stretch_prob,
        )

        if strong_real:
            strong_df = pd.read_csv(config["data"]["strong_tsv"], sep="\t")
            strong_set = StronglyAnnotatedSet(
                config["data"]["strong_folder"],
                strong_df,
                encoder,
                pad_to=config["data"]["audio_max_len"],
                feats_pipeline=feature_extraction,
                use_filter_aug=use_filter_aug,
                filter_aug_prob=filter_aug_prob,
                use_spec_aug=use_spec_aug,
                spec_aug_prob=spec_aug_prob,
                use_time_stretch=use_time_stretch,  
                time_stretch_prob=time_stretch_prob,
            )

        weak_df = pd.read_csv(config["data"]["weak_tsv"], sep="\t")
        train_weak_df = weak_df.sample(
            frac=config["training"]["weak_split"],
            random_state=config["training"]["seed"],
        )
        valid_weak_df = weak_df.drop(train_weak_df.index).reset_index(drop=True)
        train_weak_df = train_weak_df.reset_index(drop=True)
        weak_set = WeakSet(
            config["data"]["weak_folder"],
            train_weak_df,
            encoder,
            pad_to=config["data"]["audio_max_len"],
            feats_pipeline=feature_extraction,
            use_filter_aug=use_filter_aug,
            filter_aug_prob=filter_aug_prob,
            use_spec_aug=use_spec_aug,
            spec_aug_prob=spec_aug_prob,
            use_time_stretch=use_time_stretch,
            time_stretch_prob=time_stretch_prob,
        )

        unlabeled_set = UnlabeledSet(
            config["data"]["unlabeled_folder"],
            encoder,
            pad_to=config["data"]["audio_max_len"],
            feats_pipeline=feature_extraction,
            use_filter_aug=use_filter_aug,
            filter_aug_prob=filter_aug_prob,
            use_spec_aug=use_spec_aug,
            spec_aug_prob=spec_aug_prob,
            use_time_stretch=use_time_stretch,
            time_stretch_prob=time_stretch_prob,
        )

        synth_df_val = pd.read_csv(config["data"]["synth_val_tsv"], sep="\t")
        synth_val = StronglyAnnotatedSet(
            config["data"]["synth_val_folder"],
            synth_df_val,
            encoder,
            return_filename=True,
            pad_to=config["data"]["audio_max_len"],
            test=True,
        )

        weak_val = WeakSet(
            config["data"]["weak_folder"],
            valid_weak_df,
            encoder,
            pad_to=config["data"]["audio_max_len"],
            return_filename=True,
            test=True,
        )

        if strong_real:
            strong_full_set = torch.utils.data.ConcatDataset([strong_set, synth_set])
            tot_train_data = [strong_full_set, weak_set, unlabeled_set]
        else:
            tot_train_data = [synth_set, weak_set, unlabeled_set]
        train_dataset = torch.utils.data.ConcatDataset(tot_train_data)

        batch_sizes = config["training"]["batch_size"]
        samplers = [torch.utils.data.RandomSampler(x) for x in tot_train_data]
        batch_sampler = ConcatDatasetBatchSampler(samplers, batch_sizes)

        valid_dataset = torch.utils.data.ConcatDataset([synth_val, weak_val])

        ##### training params and optimizers ############
        epoch_len = min(
            [
                len(tot_train_data[indx])
                // (
                    config["training"]["batch_size"][indx]
                    * config["training"]["accumulate_batches"]
                )
                for indx in range(len(tot_train_data))
            ]
        )

        opt = torch.optim.Adam(
            sed_student.parameters(), config["opt"]["lr"], betas=(0.9, 0.999)
        )
        exp_steps = config["training"]["n_epochs_warmup"] * epoch_len
        exp_scheduler = {
            "scheduler": ExponentialWarmup(opt, config["opt"]["lr"], exp_steps),
            "interval": "step",
        }
        logger = TensorBoardLogger(
            os.path.dirname(config["log_dir"]),
            config["log_dir"].split("/")[-1],
        )
        logger.log_hyperparams(config)
        print(f"experiment dir: {logger.log_dir}")
        
        # Log augmentation settings
        print(f"\n=== Augmentation Settings ===")
        print(f"Filter Augmentation: {'ON' if use_filter_aug else 'OFF'} (prob={filter_aug_prob})")
        print(f"Spec Augmentation: {'ON' if use_spec_aug else 'OFF'} (prob={spec_aug_prob})")
        print(f"Time Stretch: {'ON' if use_time_stretch else 'OFF'} (prob={time_stretch_prob})")
        print(f"=============================\n")

        if callbacks is None:
            callbacks = [
                EarlyStopping(
                    monitor="val/obj_metric",
                    patience=config["training"]["early_stop_patience"],
                    verbose=True,
                    mode="max",
                ),
                ModelCheckpoint(
                    logger.log_dir,
                    monitor="val/obj_metric",
                    save_top_k=1,
                    mode="max",
                    save_last=True,
                ),
            ]
    else:
        train_dataset = None
        valid_dataset = None
        batch_sampler = None
        opt = None
        exp_scheduler = None
        logger = True
        callbacks = None

    desed_training = SEDTask4(
        config,
        encoder=encoder,
        sed_student=sed_student,
        opt=opt,
        train_data=train_dataset,
        valid_data=valid_dataset,
        test_data=test_dataset,
        train_sampler=batch_sampler,
        scheduler=exp_scheduler,
        fast_dev_run=fast_dev_run,
        evaluation=evaluation,
    )

    # Not using the fast_dev_run of Trainer because creates a DummyLogger so cannot check problems with the Logger
    if fast_dev_run:
        flush_logs_every_n_steps = 1
        log_every_n_steps = 1
        limit_train_batches = 2
        limit_val_batches = 2
        limit_test_batches = 2
        n_epochs = 3
    else:
        flush_logs_every_n_steps = 100
        log_every_n_steps = 40
        limit_train_batches = 1.0
        limit_val_batches = 1.0
        limit_test_batches = 1.0
        n_epochs = config["training"]["n_epochs"]

    if gpus == "0":
        accelerator = "cpu"
        devices = 1
    elif gpus == "1":
        accelerator = "gpu"
        devices = 1
    else:
        raise NotImplementedError("Multiple GPUs are currently not supported")

    trainer = pl.Trainer(
        precision=config["training"]["precision"],
        max_epochs=n_epochs,
        callbacks=callbacks,
        accelerator=accelerator,
        devices=devices,
        strategy="auto",
        accumulate_grad_batches=config["training"]["accumulate_batches"],
        logger=logger,
        gradient_clip_val=config["training"]["gradient_clip"],
        check_val_every_n_epoch=config["training"]["validation_interval"],
        num_sanity_val_steps=0,
        log_every_n_steps=log_every_n_steps,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        deterministic=config["training"]["deterministic"],
        enable_progress_bar=config["training"]["enable_progress_bar"],
    )
    if test_state_dict is None:
        # start tracking energy consumption
        trainer.fit(desed_training, ckpt_path=checkpoint_resume)
        best_path = trainer.checkpoint_callback.best_model_path
        print(f"best model: {best_path}")
        test_state_dict = torch.load(best_path)["state_dict"]

    desed_training.load_state_dict(test_state_dict)
    trainer.test(desed_training)


def prepare_run(argv=None):
    parser = argparse.ArgumentParser("Training a SED system for DESED Task")
    parser.add_argument(
        "--conf_file",
        default="./confs/default.yaml",
        help="The configuration file with all the experiment parameters.",
    )
    parser.add_argument(
        "--log_dir",
        default="./exp/2023_baseline",
        help="Directory where to save tensorboard logs, saved models, etc.",
    )
    parser.add_argument(
        "--strong_real",
        action="store_true",
        default=False,
        help="The strong annotations coming from Audioset will be included in the training phase.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Allow the training to be resumed, take as input a previously saved model (.ckpt).",
    )
    parser.add_argument(
        "--test_from_checkpoint", 
        default=None, 
        help="Test the model specified"
    )
    parser.add_argument(
        "--gpus",
        default="1",
        help="The number of GPUs to train on, or the gpu to use, default='1', "
        "so uses one GPU",
    )
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        default=False,
        help="Use this option to make a 'fake' run which is useful for development and debugging. "
        "It uses very few batches and epochs so it won't give any meaningful result.",
    )
    parser.add_argument(
        "--eval_from_checkpoint", 
        default=None, 
        help="Evaluate the model specified"
    )
    
    # Augmentation arguments
    parser.add_argument(
        "--use_filter_aug",
        action="store_true",
        default=True,
        help="Enable filter augmentation (default: True)",
    )
    parser.add_argument(
        "--no_filter_aug",
        action="store_false",
        dest="use_filter_aug",
        help="Disable filter augmentation",
    )
    parser.add_argument(
        "--filter_aug_prob",
        type=float,
        default=0.5,
        help="Probability of applying filter augmentation (default: 0.5)",
    )
    parser.add_argument(
        "--use_spec_aug",
        action="store_true",
        default=False,
        help="Enable spec augmentation (default: False)",
    )
    parser.add_argument(
        "--spec_aug_prob",
        type=float,
        default=0.5,
        help="Probability of applying spec augmentation (default: 0.5)",
    )
    parser.add_argument(
        "--use_time_stretch",
        action="store_true",
        default=True,
        help="Enable time stretch augmentation (default: True)",
    )
    parser.add_argument(
        "--no_time_stretch",
        action="store_false",
        dest="use_time_stretch",
        help="Disable time stretch augmentation",
    )
    parser.add_argument(
        "--time_stretch_prob",
        type=float,
        default=0.5,
        help="Probability of applying time stretch (default: 0.5)",
    )

    args = parser.parse_args(argv)

    with open(args.conf_file, "r") as f:
        configs = yaml.safe_load(f)

    evaluation = False
    test_from_checkpoint = args.test_from_checkpoint

    if args.eval_from_checkpoint is not None:
        test_from_checkpoint = args.eval_from_checkpoint
        evaluation = True

    test_model_state_dict = None
    if test_from_checkpoint is not None:
        checkpoint = torch.load(test_from_checkpoint)
        configs_ckpt = checkpoint["hyper_parameters"]
        configs_ckpt["data"] = configs["data"]
        print(
            f"loaded model: {test_from_checkpoint} \n"
            f"at epoch: {checkpoint['epoch']}"
        )
        test_model_state_dict = checkpoint["state_dict"]

    if evaluation:
        configs["training"]["batch_size_val"] = 1

    test_only = test_from_checkpoint is not None
    resample_data_generate_durations(configs["data"], test_only, evaluation)
    
    return configs, args, test_model_state_dict, evaluation


if __name__ == "__main__":
    # prepare run
    configs, args, test_model_state_dict, evaluation = prepare_run()

    # launch run
    single_run(
        configs,
        args.log_dir,
        args.gpus,
        args.strong_real,
        args.resume_from_checkpoint,
        test_model_state_dict,
        args.fast_dev_run,
        evaluation,
        callbacks=None,
        use_filter_aug=args.use_filter_aug,
        filter_aug_prob=args.filter_aug_prob,
        use_spec_aug=args.use_spec_aug,
        spec_aug_prob=args.spec_aug_prob,
        use_time_stretch=args.use_time_stretch,
        time_stretch_prob=args.time_stretch_prob,
    )
