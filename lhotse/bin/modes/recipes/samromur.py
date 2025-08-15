import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.samromur import download_dataset, prepare_dataset
from lhotse.utils import Pathlike

__all__ = ["samromur"]


@download.command(context_settings=dict(show_default=True))
@click.argument("dataset_name", type=click.Path(), default="samromur")
@click.argument("target_dir", type=click.Path(), default=".")
@click.option(
    "--part",
    type=click.Choice(["train", "dev", "test", "all"]),
    default="all",
    help="Which part of the dataset to download",
)
@click.option(
    "--force-download/--no-force-download",
    default=False,
    help="Whether to force download even if files exist",
)
@click.option(
    "--remove-archive/--no-remove_archive",
    default=False,
    help="Whether to remove tar files after extraction",
)
def samromur(
    target_dir: Pathlike,
    dataset_name: str,
    part: str,
    force_download: bool,
    remove_archive: bool,
):
    """
    Download Samromur or Malromur Icelandic speech datasets.

    DATASET_NAME: Which dataset to download - 'samromur' (default) or 'malromur'.

    TARGET_DIR: Directory where the dataset will be downloaded (default: current directory).

    The downloaded data includes audio files and corresponding metadata.
    """
    if dataset_name == "malromur" and part != "train":
        raise ValueError("Malromur dataset only has a 'train' part.")

    download_dataset(
        target_dir=target_dir,
        dataset_name=dataset_name,
        part=part,
        force_download=force_download,
        remove_archive=remove_archive,
    )


@prepare.command(context_settings=dict(show_default=True))
@click.argument("dataset_name", type=click.Path(), default="samromur")
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "--part",
    type=click.Choice(["train", "dev", "test", "all"]),
    default="all",
    help="Which part of the dataset to prepare",
)
def samromur(
    dataset_name: str,
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    part: str,
):
    """
    Prepare Samromur or Malromur dataset manifests for use with Lhotse.

    DATASET_NAME: Which dataset to prepare - 'samromur' (default) or 'malromur'.

    CORPUS_DIR: Path to the downloaded dataset. Should contain subdirectories
    like 'train', 'dev', 'test' for Samromur or 'train' for Malromur.

    OUTPUT_DIR: Directory where the resulting manifests will be stored.

    This command creates recording and supervision manifests in JSON format
    that can be used with Lhotse's data structures. The manifests include
    information about audio recordings and corresponding transcriptions.
    """
    prepare_dataset(
        dataset_name=dataset_name,
        corpus_dir=corpus_dir,
        manifest_dir=output_dir,
        part=part,
    )
