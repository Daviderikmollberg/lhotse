import csv
import logging
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Union, List
import re
from tqdm import tqdm

from lhotse.audio import Recording, RecordingSet
from lhotse.qa import fix_manifests, validate_recordings_and_supervisions
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, resumable_download, safe_extract


SAMROMUR_HUGGINGFACE_DATASET_URL = (
    "https://huggingface.co/datasets/DavidErikMollberg/samromur_asr/resolve/main/data"
)
MALROMUR_DATASET_URL = "https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/202/malromur.zip"


def _validate_parts(part: str, valid_parts: List[str]) -> None:
    """
    Validate the part against the list of valid parts.
    Raises ValueError if the part is not valid.
    """
    if not isinstance(part, str):
        raise ValueError(f"Part must be a string, got {type(part)} instead.")
    if part not in valid_parts:
        raise ValueError(f"Invalid part: {part}. Choose from {valid_parts}.")


def _download_samromur(
    part: str, target_dir: Pathlike, force_download: bool, remove_archive: bool
) -> Dict[str, Path]:
    """
    Download the Samromur dataset from HuggingFace.

    :param part: Which part(s) of the dataset to download ("train", "dev", "test", or "all")
    :param target_dir: Directory where the dataset will be downloaded
    :param force_download: If True, force redownload even if files exist
    :param remove_archive: If True, remove the downloaded archive files after extraction
    :return: Dictionary mapping parts to their directory paths
    """
    downloaded_parts = {}

    dataset_name = "samromur"
    _validate_parts(part, ["train", "dev", "test", "all"])
    if part == "all":
        parts = ["train", "dev", "test"]
    else:
        parts = [part]

    for p in tqdm(parts):
        part_dir = target_dir / dataset_name / p
        completed_detector = part_dir / ".completed"

        if completed_detector.is_file() and not force_download:
            logging.info(
                f"Skipping {p} part of {dataset_name} because {completed_detector} exists."
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
        audio_url = f"{SAMROMUR_HUGGINGFACE_DATASET_URL}/{p}/audio.tar.gz"

        logging.info(f"Downloading {p} audio to {audio_tar_path}")
        resumable_download(
            audio_url,
            filename=audio_tar_path,
            force_download=force_download,
        )

        # Download the metadata file
        metadata_path = part_dir / "metadata.csv"
        metadata_url = f"{SAMROMUR_HUGGINGFACE_DATASET_URL}/{p}/metadata.csv"
        logging.info(f"Downloading {p} metadata to {metadata_path}")
        resumable_download(
            metadata_url,
            filename=metadata_path,
            force_download=force_download,
        )

        with open(part_dir / "metadata.csv", "r", encoding="utf-8") as f:
            content = f.readlines()

        # The correct way to pop the first element
        header = content[0].strip().split(",")
        content = content[1:]  # Remove the first line properly
        header = content[0] + ["speaker_id"] + header[1:]

        # Write back with header
        with open(part_dir / "metadata.csv", "w", encoding="utf-8") as f:
            # Write header line first
            f.write(",".join(header) + "\n")
            # Write the original content
            for line in content:
                f.write(line)
        logging.info(f"Downloaded and prepared {part} part of {dataset_name}")

        # Extract the audio tar.gz file
        logging.info(f"Extracting {p} audio to {part_dir}")
        with tarfile.open(audio_tar_path) as tar:
            safe_extract(tar, path=part_dir)

        # Mark as completed
        completed_detector.touch()
        downloaded_parts[p] = part_dir
        logging.info(f"Downloaded and prepared {p} part of {dataset_name}")
        if remove_archive:
            audio_tar_path.unlink(missing_ok=True)
    return downloaded_parts


def _download_malromur(
    target_dir: Pathlike, force_download: bool, remove_archive: bool
) -> Dict[str, Path]:
    """
    Download the Malromur dataset from the specified URL.

    :param target_dir: Directory where the dataset will be downloaded
    :param force_download: If True, force redownload even if files exist
    :param remove_archive: If True, remove the downloaded tar.gz files after extraction
    :return: Dictionary mapping parts to their directory paths
    """
    dataset_name = "malromur"
    target_dir = Path(target_dir) / dataset_name
    target_dir.mkdir(parents=True, exist_ok=True)

    part = "train"  # Malromur does not have dev/test splits, so we use 'train' as a placeholder
    part_dir = target_dir / part

    audio_dir = part_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    completed_detector = part_dir / ".completed"

    if completed_detector.is_file() and not force_download:
        logging.info(f"Skipping Malromur download because {completed_detector} exists.")
        return {"malromur": part_dir}

    part_dir.mkdir(parents=True, exist_ok=True)

    # Download the Malromur zip file
    malromur_zip_path = part_dir / "malromur.zip"
    logging.info(f"Downloading Malromur dataset to {malromur_zip_path}")
    resumable_download(
        MALROMUR_DATASET_URL,
        filename=malromur_zip_path,
        force_download=force_download,
    )

    # Extract the zip file, only files from "correct/*" into "audio" directory
    logging.info(f"Extracting Malromur dataset to {part_dir}")

    with zipfile.ZipFile(malromur_zip_path, "r") as zip_ref:
        # Get all files that match the correct/* pattern
        correct_files = [f for f in zip_ref.namelist() if f.startswith("correct/")]

        # Extract only the files in the correct/* directory
        for file in tqdm(correct_files, desc="Extracting audio files"):
            # Extract to temporary location, preserving the relative path
            zip_ref.extract(file, path=part_dir)

        # Simply rename the "correct" directory to "audio" instead of moving files individually
        correct_dir = part_dir / "correct"
        if correct_dir.exists():
            # Remove existing audio_dir if it exists
            if audio_dir.exists():
                shutil.rmtree(audio_dir)
            # Rename the directory
            correct_dir.rename(audio_dir)

        # # Extract the metadata file
        metadata_source_file = "info.txt"
        zip_ref.extract(metadata_source_file, path=part_dir)

        # Add header to the metadata file
        header = [
            "file_name",
            "speaker_id",
            "id",
            "environment",
            "unk",
            "gender",
            "age_range",
            "text",
            "duration",
            "classification",
        ]
        logging.info(f"Extracting metadata")
        # Write back with header
        with open(part_dir / "info.txt", "r", encoding="utf-8") as f_in, open(
            part_dir / "metadata.csv", "w", encoding="utf-8"
        ) as f_out:
            # Write header line first
            f_out.write(",".join(header) + "\n")
            # Write the original content
            for line in f_in:
                # Only keep lines where the classification is "correct"
                if line.split(",")[-1].strip() != "correct":
                    continue
                id = line.split(",")[0].strip()
                audio_file_name = Path("audio") / (id + ".wav")
                speaker_id = id.split("T")[0]
                f_out.write(f"{audio_file_name},{speaker_id}," + line)

            logging.info(f"Downloaded and prepared {part} part of {dataset_name}")
        shutil.rmtree(part_dir / "info.txt", ignore_errors=True)
        # Mark as completed
        completed_detector.touch()

    if remove_archive:
        malromur_zip_path.unlink(missing_ok=True)

    return {part: part_dir}


def download_dataset(
    target_dir: Pathlike = ".",
    dataset_name: Literal["samromur", "malromur"] = "samromur",
    force_download: Optional[bool] = False,
    part: Literal["train", "dev", "test", "all"] = "all",
    remove_archive: bool = False,
) -> Dict[str, Path]:
    """
    Download the Samromur or Malromur dataset.

    :param target_dir: Directory where the dataset will be downloaded
    :param dataset_name: Which dataset to download ("samromur" or "malromur")
    :param force_download: If True, force redownload even if files exist
    :param part: Which part(s) of the dataset to download ("train", "dev", "test", or "all")
    :param remove_archive: If True, remove the downloaded archive files after extraction to save on space
    :return: Dictionary mapping parts to their directory paths
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if dataset_name == "samromur":
        logging.info("Downloading Samromur dataset")
        downloaded_parts = _download_samromur(
            part, target_dir, force_download, remove_archive
        )
    elif dataset_name == "malromur":
        logging.info("Downloading Malromur dataset")
        _validate_parts(part, ["train", "all"])
        downloaded_parts = _download_malromur(
            target_dir, force_download, remove_archive
        )
    return downloaded_parts


def _load_metadata(
    corpus_dir: Pathlike,
    part: Literal["train", "dev", "test"] = "train",
    headers: Optional[List[str]] = None,
    delimiter: str = ",",
    audio_file_key: str = "file_name",
) -> List[Dict]:
    """
    Load metadata from a dataset.

    :param corpus_dir: Directory where the dataset is stored
    :param part: Which part of the dataset to load metadata for
    :param headers: Optional list of column headers if the CSV doesn't have a header row
    :param delimiter: Character used to separate fields in the metadata file
    :param audio_file_key: The key in the metadata that contains the audio filename
    :return: List of dictionaries containing metadata for each audio file
    """
    corpus_dir = Path(corpus_dir)
    metadata_path = corpus_dir / part / "metadata.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(
            f, fieldnames=headers if headers else None, delimiter=delimiter
        )
        for row in reader:
            # Add the audio file path to the item
            row["audio_file"] = corpus_dir / part / row[audio_file_key]
            metadata.append(row)

    return metadata


def _prepare(
    metadata: List[Dict],
    text_key: str = "text",
    audio_file_path_key: str = "audio_file",
    audio_id_key: str = "id",
    gender_key: str = None,
    speaker_key: str = None,
    custom: dict = None,
) -> Tuple[List[Recording], List[SupervisionSegment]]:
    """
    Create recordings and supervision segments from metadata.

    :param metadata: List of dictionaries containing metadata for each audio file
    :param text_key: Key in metadata for the transcription text
    :param audio_file_path_key: Key in metadata for the audio file path
    :param audio_id_key: Key in metadata for the unique identifier
    :param gender_key: Key in metadata for gender information (optional)
    :param speaker_key: Key in metadata for speaker ID (optional)
    :param custom: Dictionary mapping custom keys to metadata keys to include in supervisions
    :return: Tuple of (recordings, supervisions) lists
    """
    recordings = []
    supervisions = []
    for item in tqdm(metadata, desc="Preparing recordings and supervisions"):
        recording = Recording.from_file(
            item.get(audio_file_path_key), recording_id=item.get(audio_id_key)
        )
        supervision_segment = SupervisionSegment(
            id=recording.id,
            recording_id=recording.id,
            start=0.0,
            duration=recording.duration,
            text=item.get(text_key),
            gender=item.get(gender_key),
            speaker=item.get(speaker_key),
            custom={k: item.get(v) for k, v in custom.items()} if custom else {},
        )

        recordings.append(recording)
        supervisions.append(supervision_segment)
    return recordings, supervisions


def _prepare_samromur(
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
    dataset_name = "samromur"
    _validate_parts(part, ["train", "dev", "test", "all"])

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

        metadata = _load_metadata(corpus_dir=corpus_dir, part=p)

        recordings, supervisions = _prepare(
            metadata=metadata,
            text_key="text_no_punct",
            audio_file_path_key="audio_file",
            audio_id_key="id",
            gender_key="gender",
            speaker_key="speaker_id",
            custom={
                "age": "age_range",
                "text_orginal": "text",
                "text_normalized": "text_normalized",
            },
        )

        recording_set = RecordingSet.from_recordings(recordings)
        supervision_set = SupervisionSet.from_segments(supervisions)
        recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
        validate_recordings_and_supervisions(recording_set, supervision_set)

        if manifest_dir is not None:
            supervision_set.to_file(
                manifest_dir / f"{dataset_name}_supervisions_{p}.jsonl"
            )
            recording_set.to_file(manifest_dir / f"{dataset_name}_recordings_{p}.jsonl")

        result[p] = {
            "recordings": recording_set,
            "supervisions": supervision_set,
        }

    return result


def _prepare_malromur(
    corpus_dir: Pathlike,
    manifest_dir: Optional[Pathlike] = None,
    part: Literal["train", "all"] = "all",
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Prepare the Malromur dataset by creating manifests from downloaded data.

    :param corpus_dir: Directory where the dataset has been downloaded
    :param manifest_dir: Optional directory where manifests will be stored
    :param part: Which part(s) of the dataset to prepare ("train" or "all")
    :return: Dictionary with dataset parts and their recording/supervision sets
    """
    dataset_name = "malromur"
    _validate_parts(part, ["train", "all"])

    parts_to_process = part
    if part == "all":
        parts_to_process = "train"

    logging.info(f"Preparing {parts_to_process} part of the Samromur dataset")

    if not (corpus_dir / parts_to_process).exists():
        logging.warning(
            f"Part {parts_to_process} does not exist in {corpus_dir}, skipping preparation."
        )
        return {}
    metadata = _load_metadata(corpus_dir=corpus_dir, part=parts_to_process)

    recordings, supervisions = _prepare(
        metadata=metadata,
        text_key="text",
        audio_file_path_key="audio_file",
        audio_id_key="id",
        gender_key="gender",
        speaker_key="speaker_id",
        custom={
            "environment": "Environment",
        },
    )

    recording_set = RecordingSet.from_recordings(recordings)
    supervision_set = SupervisionSet.from_segments(supervisions)
    recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
    validate_recordings_and_supervisions(recording_set, supervision_set)

    if manifest_dir is not None:
        supervision_set.to_file(
            manifest_dir / f"{dataset_name}_supervisions_{parts_to_process}.jsonl"
        )
        recording_set.to_file(
            manifest_dir / f"{dataset_name}_recordings_{parts_to_process}.jsonl"
        )

    result = {}
    result[parts_to_process] = {
        "recordings": recording_set,
        "supervisions": supervision_set,
    }

    return result


def prepare_dataset(
    dataset_name: Literal["samromur", "malromur"] = "samromur",
    corpus_dir: Pathlike = "data/samromur",
    manifest_dir: Optional[Pathlike] = "data/manifests",
    part: Literal["train", "dev", "test", "all"] = "all",
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Prepare manifests for the Samromur or Malromur dataset.

    :param dataset_name: Which dataset to prepare ("samromur" or "malromur")
    :param corpus_dir: Directory where the dataset has been downloaded
    :param manifest_dir: Optional directory where manifests will be stored
    :param part: Which part(s) of the dataset to prepare ("train", "dev", "test", or "all")
    :return: Dictionary with dataset parts and their recording/supervision sets
    """
    manifest_dir = Path(manifest_dir) if manifest_dir else None
    corpus_dir = Path(corpus_dir)
    if manifest_dir is not None:
        manifest_dir.mkdir(parents=True, exist_ok=True)

    if dataset_name == "samromur":

        manifests = _prepare_samromur(
            corpus_dir=corpus_dir,
            manifest_dir=manifest_dir,
            part=part,
        )
    elif dataset_name == "malromur":
        manifests = _prepare_malromur(
            corpus_dir=corpus_dir,
            manifest_dir=manifest_dir,
            part="train",
        )
    return manifests


# if __name__ == "__main__":
#     # First download the dataset
# download_dataset(
#     target_dir=Path("data"), dataset_name="malromur"
# )

# Then prepare the manifests
# prepare_dataset(
#     corpus_dir=Path("data/malromur"),
#     part="all",
#     dataset_name="malromur",
#     manifest_dir=Path("data/manifests"),
# )
