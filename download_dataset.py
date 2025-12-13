import argparse
import glob
import os
import shutil
import time
import warnings
from pathlib import Path
from pprint import pformat

import desed


def create_folder(folder, exist_ok=True, delete_if_exists=False):
    """Create folder (and parent folders) if not exists.

    Args:
        folder: str, path of folder(s) to create.
        delete_if_exists: bool, True if you want to delete the folder when exists

    Returns:
        None
    """
    if not folder == "":
        if delete_if_exists:
            if os.path.exists(folder):
                shutil.rmtree(folder)
                os.mkdir(folder)

        os.makedirs(folder, exist_ok=exist_ok)


def _create_symlink(src, dest, **kwargs):
    if os.path.exists(dest):
        warnings.warn(f"Symlink already exists : {dest}, skipping.\n")
    else:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        os.symlink(os.path.abspath(src), dest, **kwargs)


def create_synth_dcase(synth_path, destination_folder):
    """Create symbolic links for synethtic part of the dataset

    Args:
        synth_path (str): synthetic folder path
        destination_folder (str): destination folder path
    """
    print("Creating symlinks for synthetic data")
    split_sets = ["train", "validation"]
    if os.path.exists(os.path.join(synth_path, "audio", "eval")):
        split_sets.append("eval")

    for split_set in split_sets:
        # AUDIO
        split_audio_folder = os.path.join(synth_path, "audio", split_set)
        audio_subfolders = [
            d
            for d in os.listdir(split_audio_folder)
            if os.path.isdir(os.path.join(split_audio_folder, d))
        ]
        # Manage the validation case which changed from 2020
        if split_set == "validation" and not len(audio_subfolders):
            split_audio_folder = os.path.join(synth_path, "audio")
            audio_subfolders = ["validation"]

        for subfolder in audio_subfolders:
            abs_src_folder = os.path.abspath(
                os.path.join(split_audio_folder, subfolder)
            )
            dest_folder = os.path.join(
                destination_folder, "audio", split_set, subfolder
            )
            _create_symlink(abs_src_folder, dest_folder)

        # META
        split_meta_folder = os.path.join(
            synth_path, "metadata", split_set, f"synthetic21_{split_set}"
        )
        meta_files = glob.glob(os.path.join(split_meta_folder, "*.tsv"))
        for meta_file in meta_files:
            create_folder(destination_folder)
            dest_file = os.path.join(
                destination_folder,
                "metadata",
                split_set,
                f"synthetic21_{split_set}",
                os.path.basename(meta_file),
            )
            _create_symlink(meta_file, dest_file)


if __name__ == "__main__":
    t = time.time()

    DESED_DIR = '/home/vlad/DESED_task'
    bdir = os.path.join(DESED_DIR, 'data')

    missing_files = None

    dcase_dataset_folder = os.path.join(bdir, "dcase", "dataset")

    missing_files = desed.download_audioset_data(
        dcase_dataset_folder, n_jobs=3, chunk_size=10
    )

    url_synth = "https://zenodo.org/record/6026841/files/dcase_synth.zip?download=1"
    synth_folder = str(os.path.basename(url_synth)).split(".")[0]

    desed.download.download_and_unpack_archive(
        url_synth, dcase_dataset_folder, archive_format="zip"
    )

    synth_folder = os.path.join(bdir, "dcase", "dataset", synth_folder)
    create_synth_dcase(synth_folder, dcase_dataset_folder)
