import csv
import logging
import shutil
import tarfile
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Union, List
import re
from tqdm import tqdm

from lhotse.audio import Recording, RecordingSet
from lhotse.qa import fix_manifests, validate_recordings_and_supervisions
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, resumable_download, safe_extract


HUGGINGFACE_DATASET_URL = (
    "https://huggingface.co/datasets/DavidErikMollberg/samromur_asr/resolve/main/data"
)

DATASET_NAME = "samromur"


def download_samromur(
    target_dir: Pathlike = ".",
    force_download: Optional[bool] = False,
    part: Literal["train", "dev", "test", "all"] = "all",
    remove_tar: bool = True,
) -> Dict[str, Path]:
    """
    Download the Samromur dataset from HuggingFace.

    :param target_dir: Directory where the dataset will be downloaded
    :param force_download: If True, force redownload even if files exist
    :param part: Which part(s) of the dataset to download ("train", "dev", "test", or "all")
    :param remove_tar: If True, remove the downloaded tar.gz files after extraction
    :return: Dictionary mapping parts to their directory paths
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Determine which parts to download
    if part == "all":
        parts = ["train", "validation", "test"]
    else:
        parts = [part]

    downloaded_parts = {}

    for p in tqdm(parts):
        part_dir = target_dir / DATASET_NAME / p
        completed_detector = part_dir / ".completed"

        if completed_detector.is_file() and not force_download:
            logging.info(
                f"Skipping {p} part of {DATASET_NAME} because {completed_detector} exists."
            )
            downloaded_parts[p] = part_dir
            continue

        # Create and clear the part directory
        part_dir.mkdir(parents=True, exist_ok=True)
        if part_dir.exists():
            shutil.rmtree(part_dir, ignore_errors=True)
        part_dir.mkdir(parents=True, exist_ok=True)

        # Download the audio tar.gz file
        audio_tar_path = part_dir / f"audio.tar.gz"
        audio_url = f"{HUGGINGFACE_DATASET_URL}/{p}/audio.tar.gz"

        logging.info(f"Downloading {p} audio to {audio_tar_path}")
        resumable_download(
            audio_url,
            filename=audio_tar_path,
            force_download=force_download,
        )

        # Download the metadata file
        metadata_path = part_dir / "metadata.csv"
        metadata_url = f"{HUGGINGFACE_DATASET_URL}/{p}/metadata.csv"
        logging.info(f"Downloading {p} metadata to {metadata_path}")
        resumable_download(
            metadata_url,
            filename=metadata_path,
            force_download=force_download,
        )

        # Extract the audio tar.gz file
        logging.info(f"Extracting {p} audio to {part_dir}")
        with tarfile.open(audio_tar_path) as tar:
            safe_extract(tar, path=part_dir)

        # Mark as completed
        completed_detector.touch()
        downloaded_parts[p] = part_dir
        logging.info(f"Downloaded and prepared {p} part of {DATASET_NAME}")
        if remove_tar:
            audio_tar_path.unlink(missing_ok=True)
    return downloaded_parts


def load_metadata(
    corpus_dir: Pathlike,
    part: Literal["train", "validation", "test"] = "train",
) -> List[Dict]:
    """
    Load metadata from the Samromur dataset.

    :param corpus_dir: Directory where the dataset is stored
    :param part: Which part of the dataset to load metadata for
    :return: List of dictionaries containing metadata for each audio file
    """
    corpus_dir = Path(corpus_dir)
    metadata_path = corpus_dir / part / "metadata.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Add the audio file path to the item
            row["audio_file"] = corpus_dir / part / row["file_name"]
            metadata.append(row)

    return metadata


def _prepare(
    metadata: List[Dict],
) -> Tuple[List[Recording], List[SupervisionSegment]]:
    """
    Create recordings and supervision segments from metadata.

    :param metadata: List of dictionaries containing metadata for each audio file
    :return: Tuple of (recordings, supervisions) lists
    """
    recordings = []
    supervisions = []
    for item in tqdm(metadata):
        recording = Recording.from_file(
            item.get("audio_file"), recording_id=item.get("id")
        )
        supervision_segment = SupervisionSegment(
            id=recording.id,
            recording_id=recording.id,
            start=0.0,
            duration=recording.duration,
            text=item.get("text_no_punct"),
            gender=item.get("gender"),
            speaker=re.sub("-.*", "", recording.id),
            custom={
                "age": item.get("age_range"),
                "text_orginal": item.get("text"),
                "text_normalized": item.pop("text_normalized"),
            },
        )

        recordings.append(recording)
        supervisions.append(supervision_segment)
    return recordings, supervisions


def prepare_samromur(
    corpus_dir: Pathlike,
    manifest_dir: Optional[Pathlike] = None,
    part: Literal["train", "dev", "test", "all"] = "all",
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Prepare the Samromur dataset by creating manifests from downloaded data.

    :param corpus_dir: Directory where the dataset has been downloaded
    :param manifest_dir: Optional directory where manifests will be stored
    :param part: Which part(s) of the dataset to prepare ("train", "dev", "test", or "all")
    :return: Dictionary with dataset parts and their recording/supervision sets
    """
    manifest_dir = Path(manifest_dir) if manifest_dir else None
    corpus_dir = Path(corpus_dir)
    if manifest_dir is not None:
        manifest_dir.mkdir(parents=True, exist_ok=True)

    # Determine which parts to process
    if part == "all":
        parts_to_process = ["train", "dev", "test"]
    else:
        parts_to_process = [part]

    result = {}
    for p in parts_to_process:
        logging.info(f"Preparing {p} part of the Samromur dataset")

        if not (corpus_dir / p).exists():
            logging.warning(
                f"Part {p} does not exist in {corpus_dir}, skipping preparation."
            )
            continue

        metadata = load_metadata(corpus_dir=corpus_dir, part=p)

        recordings, supervisions = _prepare(metadata=metadata)

        recording_set = RecordingSet.from_recordings(recordings)
        supervision_set = SupervisionSet.from_segments(supervisions)
        recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
        validate_recordings_and_supervisions(recording_set, supervision_set)

        if manifest_dir is not None:
            supervision_set.to_file(manifest_dir / f"samromur_supervisions_{p}.jsonl")
            recording_set.to_file(manifest_dir / f"samromur_recordings_{p}.jsonl")

        result[p] = {
            "recordings": recording_set,
            "supervisions": supervision_set,
        }

    return result


# Example usage
# if __name__ == "__main__":
#     # First download the dataset
#     download_samromur(
#         target_dir=Path("data"),
#         part="test",
#     )

#     # Then prepare the manifests
#     prepare_samromur(
#         corpus_dir=Path("data/samromur"),
#         part="test",
#         manifest_dir=Path("data/manifests"),
#     )
