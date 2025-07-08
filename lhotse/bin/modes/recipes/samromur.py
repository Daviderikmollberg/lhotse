import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.samromur import download_samromur, prepare_samromur
from lhotse.utils import Pathlike

__all__ = ["samromur"]


@download.command(context_settings=dict(show_default=True))
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
    "--remove-tar/--no-remove-tar",
    default=True,
    help="Whether to remove tar files after extraction",
)
def samromur(target_dir: Pathlike, part: str, force_download: bool, remove_tar: bool):
    """Samromur dataset download."""
    download_samromur(
        target_dir=target_dir,
        part=part,
        force_download=force_download,
        remove_tar=remove_tar,
    )


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "--part",
    type=click.Choice(["train", "dev", "test", "all"]),
    default="all",
    help="Which part of the dataset to prepare",
)
def samromur(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    part: str,
):
    """Samromur dataset preparation."""
    prepare_samromur(
        corpus_dir=corpus_dir,
        manifest_dir=output_dir,
        part=part,
    )
