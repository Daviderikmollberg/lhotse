import csv
import logging
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Union, List
import re
from tqdm import tqdm
import glob

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


def _download_althingi(
    part: str, target_dir: Pathlike, force_download: bool, remove_archive: bool
) -> Dict[str, Path]:
    """
    Download the Althingi dataset from Hugging Face.

    :param part: Which part(s) of the dataset to download ("train", "dev", "test", or "all")
    :param target_dir: Directory where the dataset will be downloaded
    :param force_download: If True, force redownload even if files exist
    :param remove_archive: If True, remove the downloaded archive files after extraction
    :return: Dictionary mapping parts to their directory paths
    """
    downloaded_parts = {}
    dataset_name = "althingi"

    _validate_parts(part, ["train", "dev", "test", "all"])
    if part == "all":
        parts = ["train", "dev", "test"]
    else:
        parts = [part]

    base_url = "https://huggingface.co/datasets/language-and-voice-lab/althingi_asr/resolve/main"
    repo_base_url = "https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/277"

    # First, download the original text archive to get the unnormalized text
    target_dir = Path(target_dir)
    althingi_dir = target_dir / dataset_name
    althingi_dir.mkdir(parents=True, exist_ok=True)

    # Check if we already have the text archive extracted
    original_text_dir = althingi_dir / "malfong"
    original_text_completed = original_text_dir / ".completed"

    if not original_text_completed.is_file() or force_download:
        original_text_dir.mkdir(parents=True, exist_ok=True)
        if original_text_dir.exists():
            shutil.rmtree(original_text_dir, ignore_errors=True)
        original_text_dir.mkdir(parents=True, exist_ok=True)

        # Download the original text archive
        original_text_archive = althingi_dir / "althingi_texti.tar.gz"
        original_text_url = f"{repo_base_url}/althingi_texti.tar.gz"

        logging.info(f"Downloading original text archive to {original_text_archive}")
        resumable_download(
            original_text_url,
            filename=original_text_archive,
            force_download=force_download,
        )

        # Extract the text archive
        logging.info("Extracting original text archive...")
        with tarfile.open(original_text_archive) as tar:
            safe_extract(tar, path=althingi_dir)

        # Mark as completed
        original_text_completed.touch()

        if remove_archive:
            original_text_archive.unlink(missing_ok=True)

    # Now process each part
    for p in tqdm(parts, desc="Downloading Althingi parts"):
        part_dir = althingi_dir / p
        completed_detector = part_dir / ".completed"

        # if completed_detector.is_file() and not force_download:
        #     logging.info(
        #         f"Skipping {p} part of {dataset_name} because {completed_detector} exists."
        #     )
        #     downloaded_parts[p] = part_dir
        #     continue

        # Create and clear the part directory
        part_dir.mkdir(parents=True, exist_ok=True)

        # Copy the original text files for this part
        orig_part_files_dir = original_text_dir / p
        if orig_part_files_dir.exists():
            for file in ["spk2gender", "text"]:
                src_file = orig_part_files_dir / file
                if src_file.exists():
                    dst_file = part_dir / file
                    shutil.copy2(src_file, dst_file)
                    logging.info(f"Copied original {file} for part {p}")
        else:
            logging.warning(
                f"Original text directory for part {p} not found: {orig_part_files_dir}"
            )

        # Download the metadata file from Hugging Face
        metadata_path = part_dir / "metadata.tsv"
        metadata_url = f"{base_url}/corpus/files/metadata_{p}.tsv"
        logging.info(f"Downloading {p} metadata to {metadata_path}")
        resumable_download(
            metadata_url,
            filename=metadata_path,
            force_download=force_download,
        )

        # Create audio directory
        audio_dir = part_dir / "audio"
        audio_dir.mkdir(exist_ok=True)

        # Download audio files
        if p == "train":
            # Train data is split into multiple files
            for i in tqdm(range(1, 14), desc=f"Downloading {p} audio parts"):
                part_num = f"{i:03d}"  # Format to 001, 002, etc.
                audio_tar_path = part_dir / f"train_part_{part_num}.tar.gz"
                audio_url = (
                    f"{base_url}/corpus/speech/train/train_part_{part_num}.tar.gz"
                )

                logging.info(
                    f"Downloading train part {part_num} audio to {audio_tar_path}"
                )
                resumable_download(
                    audio_url,
                    filename=audio_tar_path,
                    force_download=force_download,
                )

                # Extract the audio tar.gz file
                logging.info(f"Extracting train part {part_num} audio to {part_dir}")
                with tarfile.open(audio_tar_path) as tar:
                    safe_extract(tar, path=part_dir)

                if remove_archive:
                    audio_tar_path.unlink(missing_ok=True)
        else:
            # Dev and test are single files
            audio_tar_path = part_dir / f"{p}.tar.gz"
            audio_url = f"{base_url}/corpus/speech/{p}.tar.gz"

            logging.info(f"Downloading {p} audio to {audio_tar_path}")
            resumable_download(
                audio_url,
                filename=audio_tar_path,
                force_download=force_download,
            )

            # Extract the audio tar.gz file
            logging.info(f"Extracting {p} audio to {part_dir}")
            with tarfile.open(audio_tar_path) as tar:
                safe_extract(tar, path=part_dir)

            if remove_archive:
                audio_tar_path.unlink(missing_ok=True)

        # Find all flac files in the specified directory structure
        flac_files = glob.glob(str(part_dir / p / "*" / "*" / "*.flac"))
        logging.info(f"Found {len(flac_files)} FLAC files to process")

        for flac_file in tqdm(flac_files, desc=f"Processing {p} audio files"):
            flac_path = Path(flac_file)
            prefixed_filename = f"{flac_path.parts[-3]}-{flac_path.name}"
            destination = audio_dir / prefixed_filename
            shutil.move(flac_file, destination)

        logging.info(
            f"Processed and moved {len(flac_files)} audio files to {audio_dir}"
        )

        # Create a metadata.csv file with the appropriate format for our processing functions
        # Merge the normalized text from Hugging Face with the original text
        _create_merged_metadata_for_althingi_data(part_dir, metadata_path)

        # Mark as completed
        completed_detector.touch()
        downloaded_parts[p] = part_dir
        logging.info(f"Downloaded and prepared {p} part of {dataset_name}")

    return downloaded_parts


def _create_merged_metadata_for_althingi_data(part_dir: Path, metadata_path: Path):
    """
    Create a merged metadata file that includes both normalized text from HuggingFace
    and original text from the repository archive.

    :param part_dir: Directory containing the downloaded part
    :param metadata_path: Path to the metadata file from HuggingFace
    """
    # Load the original text file which contains the original transcripts
    original_text_file = part_dir / "text"
    original_texts = {}

    if original_text_file.exists():
        with open(original_text_file, "r", encoding="utf-8") as f:
            for line in f:
                utterance_id = line.strip().split(" ")[0]
                text = " ".join(line.strip().split(" ")[1:])
                utterance_id_without_spk = utterance_id.split("-")[1]
                original_texts[utterance_id_without_spk] = {
                    "utterance_id": utterance_id,
                    "text": text,
                }
        logging.info(f"Loaded {len(original_texts)} original texts")
    else:
        logging.warning(f"Original text file not found: {original_text_file}")

    # Load speaker gender information
    gender_file = part_dir / "spk2gender"
    speaker_genders = {}

    if gender_file.exists():
        with open(gender_file, "r", encoding="utf-8") as f:
            for line in f:
                speaker_id, gender = line.rstrip().split("\t")
                if gender == "m":
                    gender = "male"
                elif gender == "f":
                    gender = "female"
                else:
                    gender = "unkown"
                speaker_genders[speaker_id] = gender
        logging.info(f"Loaded {len(speaker_genders)} speaker gender mappings")
    else:
        logging.warning(f"Speaker gender file not found: {gender_file}")

    # Create the merged metadata file
    output_file = part_dir / "metadata.csv"

    headers = [
        "file_name",
        "speaker_id",
        "id",
        "gender",
        "duration",
        "text",
        "normalized_text",
    ]

    with open(metadata_path, "r", encoding="utf-8") as f_in, open(
        output_file, "w", encoding="utf-8"
    ) as f_out:
        # Add our required headers
        f_out.write(",".join(headers) + "\n")

        # Skip the header line in the input
        next(f_in)

        for line in f_in:
            fields = line.strip().split("\t")
            if len(fields) < 4:
                continue

            audio_id_without_spk, _, duration, normalized_text = fields
            audio_id_with_spk = original_texts[audio_id_without_spk]["utterance_id"]
            speaker_id = audio_id_with_spk.split("-")[0]
            # Get the original text if available
            original_text = original_texts[audio_id_without_spk]["text"]

            # Get gender if available
            gender = speaker_genders.get(speaker_id)

            # Format file_name to point to the extracted audio file
            file_name = (
                f"audio/{original_texts[audio_id_without_spk]["utterance_id"]}.flac"
            )

            # Write the merged metadata
            f_out.write(
                f"{file_name},{speaker_id},{audio_id_with_spk},{gender},{duration},{original_text},{normalized_text}\n"
            )

    logging.info(f"Created merged metadata file at {output_file}")


def download_dataset(
    target_dir: Pathlike = ".",
    dataset_name: Literal["samromur", "malromur", "althingi"] = "samromur",
    force_download: Optional[bool] = False,
    part: Literal["train", "dev", "test", "all"] = "all",
    remove_archive: bool = False,
) -> Dict[str, Path]:
    """
    Download the Samromur, Malromur, or Althingi dataset.

    :param target_dir: Directory where the dataset will be downloaded
    :param dataset_name: Which dataset to download ("samromur", "malromur", or "althingi")
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
    elif dataset_name == "althingi":
        logging.info("Downloading Althingi dataset")
        downloaded_parts = _download_althingi(
            part, target_dir, force_download, remove_archive
        )
    return downloaded_parts


def _load_metadata(
    corpus_dir: Pathlike,
    part: Literal["train", "dev", "test"] = "train",
    headers: Optional[List[str]] = None,
    delimiter: str = ",",
    audio_file_key: str = "file_name",
    audio_file_prefix: str = None,
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
            if audio_file_prefix:
                # If prefix is provided, join it with the audio file path
                audio_path = audio_file_prefix / row[audio_file_key]
                row["audio_file"] = corpus_dir / part / audio_path
            else:
                # If no prefix, just use the path from the metadata
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
download_dataset(
    target_dir=Path("/home/dem/projects/k2/data"), dataset_name="althingi", part="dev"
)

# Then prepare the manifests
# prepare_dataset(
#     corpus_dir=Path("data/malromur"),
#     part="all",
#     dataset_name="malromur",
#     manifest_dir=Path("data/manifests"),
# )
