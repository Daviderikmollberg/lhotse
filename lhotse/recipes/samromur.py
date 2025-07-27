import csv
import logging
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Union, List
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
SAMROMUR_QUERIES_URL = "https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/180/samromur_queries_21.12.zip"
SAMROMUR_CHILDREN_FILES = {
    "dev": "https://huggingface.co/datasets/language-and-voice-lab/samromur_children/resolve/main/corpus/speech/dev.tar.gz",
    "test": "https://huggingface.co/datasets/language-and-voice-lab/samromur_children/resolve/main/corpus/speech/test.tar.gz",
    "train": [
        "https://huggingface.co/datasets/language-and-voice-lab/samromur_children/resolve/main/corpus/speech/train/train_part_01.tar.gz",
        "https://huggingface.co/datasets/language-and-voice-lab/samromur_children/resolve/main/corpus/speech/train/train_part_02.tar.gz",
        "https://huggingface.co/datasets/language-and-voice-lab/samromur_children/resolve/main/corpus/speech/train/train_part_03.tar.gz",
    ],
}
METADATA_URL_BASE = "https://huggingface.co/datasets/DavidErikMollberg/asr_metadata/resolve/main/{dataset}_metadata_{part}.csv"

# Configuration for different datasets
DATASET_CONFIGS = {
    "samromur": {
        "valid_parts": ["train", "dev", "test", "all"],
        "text_key": "proper_nouns_cased",
        "custom": {
            "age": "age_range",
            "text_orginal": "text",
            "text_normalized": "text_normalized",
        },
    },
    "malromur": {
        "valid_parts": ["train"],
        "text_key": "proper_nouns_cased",
        "custom": {
            "environment": "Environment",
            "text_normalized": "text_normalized",
            "text_original": "text",
        },
    },
    "althingi": {
        "valid_parts": ["train", "dev", "test", "all"],
        "text_key": "text",
        "custom": {
            "original_text": "text",
        },
    },
    "samromur_queries": {
        "valid_parts": ["train", "dev", "test", "all"],
        "text_key": "proper_nouns_cased",
        "custom": {
            "age": "age_range",
            "text_orginal": "text",
            "text_normalized": "text_normalized",
        },
    },
    "samromur_children": {
        "valid_parts": ["train", "dev", "test", "all"],
        "text_key": "proper_nouns_cased",
        "custom": {
            "age": "age_range",
            "text_orginal": "text",
            "text_normalized": "text_normalized",
        },
    },
}


def _validate_parts(part: str, valid_parts: List[str]) -> List[str]:
    """
    Validate the part against the list of valid parts.
    Raises ValueError if the part is not valid.
    """
    if not isinstance(part, str):
        raise ValueError(f"Part must be a string, got {type(part)} instead.")

    if part not in valid_parts:
        raise ValueError(f"Invalid part: {part}. Choose from {valid_parts}.")

    if part == "all":
        return [p for p in valid_parts if p != "all"]
    else:
        return [part]


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
    parts = _validate_parts(part, ["train", "dev", "test", "all"])

    for p in tqdm(parts):
        part_dir = target_dir / dataset_name / p
        completed_detector = part_dir / ".download_completed"

        if completed_detector.is_file() and not force_download:
            logging.info(
                f"Skipping {p} part of {dataset_name} because {completed_detector} exists."
            )
            downloaded_parts[p] = part_dir
            continue

        part_dir.mkdir(parents=True, exist_ok=True)

        audio_tar_path = part_dir / f"audio.tar.gz"
        audio_url = f"{SAMROMUR_HUGGINGFACE_DATASET_URL}/{p}/audio.tar.gz"

        logging.info(f"Downloading {p} audio to {audio_tar_path}")
        resumable_download(
            audio_url,
            filename=audio_tar_path,
            force_download=force_download,
        )

        metadata_url = METADATA_URL_BASE.format(part=p, dataset=dataset_name)
        metadata_path = part_dir / "metadata.csv"
        if not part_dir.exists():
            part_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Downloading {p} metadata to {metadata_path}")
        resumable_download(
            metadata_url,
            filename=metadata_path,
            force_download=force_download,
        )

        logging.info(f"Extracting {p} audio to {part_dir}")
        with tarfile.open(audio_tar_path) as tar:
            safe_extract(tar, path=part_dir)
        logging.info(f"Downloaded and prepared {part} part of {dataset_name}")

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

    part = "train"  # Malromur does not have dev/test splits, so we use 'train' as a placeholder
    part_dir = target_dir / part

    audio_dir = part_dir / "audio"
    audio_dir.mkdir(exist_ok=True, parents=True)

    completed_detector = part_dir / ".download_completed"

    if completed_detector.is_file() and not force_download:
        logging.info(f"Skipping Malromur download because {completed_detector} exists.")
        return {"malromur": part_dir}

    part_dir.mkdir(parents=True, exist_ok=True)

    malromur_zip_path = part_dir / "malromur.zip"
    logging.info(f"Downloading Malromur dataset to {malromur_zip_path}")
    resumable_download(
        MALROMUR_DATASET_URL,
        filename=malromur_zip_path,
        force_download=force_download,
    )

    metadata_url = METADATA_URL_BASE.format(part=part, dataset=dataset_name)
    metadata_path = part_dir / "metadata.csv"
    if not part_dir.exists():
        part_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Downloading {part} metadata to {metadata_path}")
    resumable_download(
        metadata_url,
        filename=metadata_path,
        force_download=force_download,
    )

    completed_detector.touch()

    if remove_archive:
        malromur_zip_path.unlink(missing_ok=True)

    return {part: part_dir}


def _download_althingi(
    part: str, target_dir: Pathlike, force_download: bool, remove_archive: bool
) -> Dict[str, Path]:
    """
    Download the Althingi dataset from Hugging Face.

    :param part: Which part(s) of the dataset to download ("train", "dev", "test", "all")
    :param target_dir: Directory where the dataset will be downloaded
    :param force_download: If True, force redownload even if files exist
    :param remove_archive: If True, remove the downloaded archive files after extraction
    :return: Dictionary mapping parts to their directory paths
    """
    downloaded_parts = {}
    dataset_name = "althingi"

    parts = _validate_parts(part, ["train", "dev", "test", "all"])

    base_url = "https://huggingface.co/datasets/language-and-voice-lab/althingi_asr/resolve/main"
    repo_base_url = "https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/277"

    target_dir = Path(target_dir)
    althingi_dir = target_dir / dataset_name
    althingi_dir.mkdir(parents=True, exist_ok=True)

    original_text_dir = althingi_dir / "malfong"
    original_text_completed = original_text_dir / ".download_completed"

    if not original_text_completed.is_file() or force_download:
        original_text_dir.mkdir(parents=True, exist_ok=True)
        if original_text_dir.exists():
            shutil.rmtree(original_text_dir, ignore_errors=True)
        original_text_dir.mkdir(parents=True, exist_ok=True)

        original_text_archive = althingi_dir / "althingi_texti.tar.gz"
        original_text_url = f"{repo_base_url}/althingi_texti.tar.gz"

        logging.info(f"Downloading original text archive to {original_text_archive}")
        resumable_download(
            original_text_url,
            filename=original_text_archive,
            force_download=force_download,
        )

        logging.info("Extracting original text archive...")
        with tarfile.open(original_text_archive) as tar:
            safe_extract(tar, path=althingi_dir)

        original_text_completed.touch()

    for p in tqdm(parts, desc="Downloading Althingi parts"):
        part_dir = althingi_dir / p
        completed_detector = part_dir / ".download_completed"

        if completed_detector.is_file() and not force_download:
            logging.info(
                f"Skipping {p} part of {dataset_name} because {completed_detector} exists."
            )
            downloaded_parts[p] = part_dir
            continue

        part_dir.mkdir(parents=True, exist_ok=True)

        if p == "test":
            orig_part_files_dir = original_text_dir / "eval"
        else:
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

        metadata_path = part_dir / "metadata.tsv"
        metadata_url = f"{base_url}/corpus/files/metadata_{p}.tsv"
        logging.info(f"Downloading {p} metadata to {metadata_path}")
        resumable_download(
            metadata_url,
            filename=metadata_path,
            force_download=force_download,
        )

        audio_dir = part_dir / "audio"
        audio_dir.mkdir(exist_ok=True)

        if p == "train":
            for i in tqdm(range(1, 14), desc=f"Downloading {p} audio parts"):
                part_num = f"{i:03d}"
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

                logging.info(f"Extracting train part {part_num} audio to {part_dir}")
                with tarfile.open(audio_tar_path) as tar:
                    safe_extract(tar, path=part_dir)

                if remove_archive:
                    audio_tar_path.unlink(missing_ok=True)

            flac_files = glob.glob(
                str(part_dir / f"train_part_*" / "*" / "*" / "*.flac")
            )
        else:
            audio_tar_path = part_dir / f"{p}.tar.gz"
            audio_url = f"{base_url}/corpus/speech/{p}.tar.gz"

            logging.info(f"Downloading {p} audio to {audio_tar_path}")
            resumable_download(
                audio_url,
                filename=audio_tar_path,
                force_download=force_download,
            )

            logging.info(f"Extracting {p} audio to {part_dir}")
            with tarfile.open(audio_tar_path) as tar:
                safe_extract(tar, path=part_dir)

            if remove_archive:
                audio_tar_path.unlink(missing_ok=True)

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
        if p == "train":
            for part_num in range(1, 14):
                part_pattern = f"train_part_{part_num:03d}"
                shutil.rmtree(part_dir / part_pattern, ignore_errors=True)
        else:

            shutil.rmtree(part_dir / p, ignore_errors=True)
        _create_merged_metadata_for_althingi_data(part_dir, metadata_path)
        completed_detector.touch()
        downloaded_parts[p] = part_dir
        logging.info(f"Downloaded and prepared {p} part of {dataset_name}")
        if remove_archive:
            shutil.rmtree(althingi_dir / "malfong", ignore_errors=True)
    if remove_archive:
        original_text_archive.unlink(missing_ok=True)

    return downloaded_parts


def _create_merged_metadata_for_althingi_data(part_dir: Path, metadata_path: Path):
    """
    Create a merged metadata file that includes both normalized text from HuggingFace
    and original text from the repository archive.

    :param part_dir: Directory containing the downloaded part
    :param metadata_path: Path to the metadata file from HuggingFace
    """
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
        f_out.write(",".join(headers) + "\n")

        next(f_in)

        for line in f_in:
            fields = line.strip().split("\t")
            if len(fields) < 4:
                continue

            audio_id_without_spk, _, duration, normalized_text = fields
            audio_id_with_spk = original_texts[audio_id_without_spk]["utterance_id"]
            speaker_id = audio_id_with_spk.split("-")[0]
            original_text = original_texts[audio_id_without_spk]["text"]
            gender = speaker_genders.get(speaker_id)
            file_name = (
                f"audio/{original_texts[audio_id_without_spk]["utterance_id"]}.flac"
            )
            f_out.write(
                f"{file_name},{speaker_id},{audio_id_with_spk},{gender},{duration},{original_text},{normalized_text}\n"
            )

    logging.info(f"Created merged metadata file at {output_file}")


def _download_samromur_queries(
    part: str, target_dir: Pathlike, force_download: bool, remove_archive: bool
) -> Dict[str, Path]:
    """
    Download the Samromur Queries dataset from the CLARIN repository.

    :param part: Which part(s) of the dataset to download ("train", "dev", "test", "all")
    :param target_dir: Directory where the dataset will be downloaded
    :param force_download: If True, force redownload even if files exist
    :param remove_archive: If True, remove the downloaded archive files after extraction
    :return: Dictionary mapping parts to their directory paths
    """
    downloaded_parts = {}
    dataset_name = "samromur_queries"
    parts = _validate_parts(part, ["train", "dev", "test", "all"])

    target_dir = Path(target_dir)
    dataset_dir = target_dir / dataset_name
    archive_path = dataset_dir / "samromur_queries_21.12.zip"

    dataset_dir.mkdir(parents=True, exist_ok=True)

    completed_detector = dataset_dir / ".download_completed"

    if not completed_detector.is_file() or force_download:
        logging.info(f"Downloading Samromur Queries dataset to {archive_path}")
        resumable_download(
            SAMROMUR_QUERIES_URL,
            filename=archive_path,
            force_download=force_download,
        )

        logging.info(f"Extracting Samromur Queries dataset to {dataset_dir}")
        full_metadata_path = dataset_dir / "metadata.tsv"
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            file_content = zip_ref.read("samromur_queries_21.12/metadata.tsv")
            with open(full_metadata_path, "wb") as f_out:
                f_out.write(file_content)
            for p in parts:
                part_dir = dataset_dir / p
                part_dir.mkdir(parents=True, exist_ok=True)

                audio_dir = part_dir / "audio"
                audio_dir.mkdir(parents=True, exist_ok=True)

                part_files = [
                    f
                    for f in zip_ref.namelist()
                    if f.startswith(f"samromur_queries_21.12/{p}/")
                ]
                if not part_files:
                    logging.warning(f"No files found for part {p} in the archive.")
                    continue

                logging.info(f"Extracting files for part {p}")
                for file in tqdm(part_files, desc=f"Extracting {p} files"):
                    if file.endswith(".flac"):
                        filename = Path(file).name
                        file_content = zip_ref.read(file)
                        output_path = audio_dir / filename
                        with open(output_path, "wb") as f_out:
                            f_out.write(file_content)

                logging.info(f"Extracted audio files to {audio_dir}")

        logging.info(f"Extracted Samromur Queries dataset to {dataset_dir}")

        if remove_archive:
            archive_path.unlink(missing_ok=True)
    else:
        logging.info(f"Skipping download as {completed_detector} exists.")

    for p in parts:
        part_dir = dataset_dir / p

        metadata_url = METADATA_URL_BASE.format(part=p, dataset=dataset_name)
        metadata_path = part_dir / "metadata.csv"
        if not part_dir.exists():
            part_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Downloading {p} metadata to {metadata_path}")
        resumable_download(
            metadata_url,
            filename=metadata_path,
            force_download=force_download,
        )

        downloaded_parts[p] = part_dir
        logging.info(f"Prepared {p} part of {dataset_name}")
        completed_detector.touch()

    return downloaded_parts


def _download_samromur_children(
    part: str, target_dir: Pathlike, force_download: bool, remove_archive: bool
) -> Dict[str, Path]:
    """
    Download the Samromur Children dataset from the specified URLs.
    :param part: Which part(s) of the dataset to download ("train", "dev", "test", or "all")
    :param target_dir: Directory where the dataset will be downloaded
    :param force_download: If True, force redownload even if files exist
    :param remove_archive: If True, remove the downloaded archive files after extraction
    :return: Dictionary mapping parts to their directory paths
    """
    downloaded_parts = {}
    dataset_name = "samromur_children"
    parts = _validate_parts(part, ["train", "dev", "test", "all"])

    target_dir = Path(target_dir)
    dataset_dir = target_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    for p in tqdm(parts, desc="Downloading Samromur Children parts"):
        part_dir = dataset_dir / p
        completed_detector = part_dir / ".download_completed"

        if completed_detector.is_file() and not force_download:
            logging.info(
                f"Skipping {p} part of {dataset_name} because {completed_detector} exists."
            )
            downloaded_parts[p] = part_dir
            continue

        part_dir.mkdir(parents=True, exist_ok=True)
        audio_dir = part_dir / "audio"
        audio_dir.mkdir(exist_ok=True)

        metadata_url = METADATA_URL_BASE.format(part=p, dataset=dataset_name)
        metadata_path = part_dir / "metadata.csv"
        logging.info(f"Downloading {p} metadata to {metadata_path}")
        resumable_download(
            metadata_url,
            filename=metadata_path,
            force_download=force_download,
        )

        if p == "train":
            for idx, url in enumerate(SAMROMUR_CHILDREN_FILES[p], 1):
                part_num = f"{idx:02d}"
                audio_tar_path = part_dir / f"train_part_{part_num}.tar.gz"
                logging.info(
                    f"Downloading train part {part_num} audio to {audio_tar_path}"
                )
                resumable_download(
                    url,
                    filename=audio_tar_path,
                    force_download=force_download,
                )

                logging.info(f"Extracting train part {part_num} audio to {part_dir}")
                with tarfile.open(audio_tar_path) as tar:
                    safe_extract(tar, path=part_dir)

                if remove_archive:
                    audio_tar_path.unlink(missing_ok=True)

            extracted_files = list(part_dir.glob("**/*.flac"))
            for flac_file in tqdm(extracted_files, desc=f"Moving {p} audio files"):
                shutil.move(str(flac_file), str(audio_dir / flac_file.name))

            shutil.rmtree(part_dir / "train_part_01", ignore_errors=True)
            shutil.rmtree(part_dir / "train_part_02", ignore_errors=True)
            shutil.rmtree(part_dir / "train_part_03", ignore_errors=True)
        else:
            audio_tar_path = part_dir / f"{p}.tar.gz"
            logging.info(f"Downloading {p} audio to {audio_tar_path}")
            resumable_download(
                SAMROMUR_CHILDREN_FILES[p],
                filename=audio_tar_path,
                force_download=force_download,
            )

            logging.info(f"Extracting {p} audio to {part_dir}")
            with tarfile.open(audio_tar_path) as tar:
                safe_extract(tar, path=part_dir)

            extracted_files = list(part_dir.glob("**/*.flac"))
            for flac_file in tqdm(extracted_files, desc=f"Moving {p} audio files"):
                shutil.move(str(flac_file), str(audio_dir / flac_file.name))

            if remove_archive:
                audio_tar_path.unlink(missing_ok=True)
            shutil.rmtree(part_dir / p, ignore_errors=True)
        logging.info(f"Processed and moved audio files for {p} part of {dataset_name}")

        completed_detector.touch()
        downloaded_parts[p] = part_dir
        logging.info(f"Downloaded and prepared {p} part of {dataset_name}")

    return downloaded_parts


def download_dataset(
    target_dir: Pathlike = ".",
    dataset_name: Literal[
        "samromur", "malromur", "althingi", "samromur_queries", "samromur_children"
    ] = "samromur",
    force_download: Optional[bool] = False,
    part: Literal["train", "dev", "test", "all"] = "all",
    remove_archive: bool = False,
) -> Dict[str, Path]:
    """
    Download the Samromur, Malromur, Althingi, Samromur Queries, or Samromur Children dataset.

    :param target_dir: Directory where the dataset will be downloaded
    :param dataset_name: Which dataset to download ("samromur", "malromur", "althingi", "samromur_queries", or "samromur_children")
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
        _validate_parts(part, ["train"])
        downloaded_parts = _download_malromur(
            target_dir, force_download, remove_archive
        )
    elif dataset_name == "althingi":
        logging.info("Downloading Althingi dataset")
        downloaded_parts = _download_althingi(
            part, target_dir, force_download, remove_archive
        )
    elif dataset_name == "samromur_queries":
        logging.info("Downloading Samromur Queries dataset")
        downloaded_parts = _download_samromur_queries(
            part, target_dir, force_download, remove_archive
        )
    elif dataset_name == "samromur_children":
        logging.info("Downloading Samromur Children dataset")
        downloaded_parts = _download_samromur_children(
            part, target_dir, force_download, remove_archive
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

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
            if audio_file_prefix:
                audio_path = audio_file_prefix / row[audio_file_key]
                row["audio_file"] = corpus_dir / part / audio_path
            else:

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


def _prepare_dataset(
    corpus_dir: Pathlike,
    manifest_dir: Optional[Pathlike] = None,
    dataset_name: str = "samromur",
    part: Literal["train", "dev", "test", "all"] = "all",
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Prepare any supported dataset by creating manifests from downloaded data.

    :param corpus_dir: Directory where the dataset has been downloaded
    :param manifest_dir: Optional directory where manifests will be stored
    :param dataset_name: Name of the dataset to prepare
    :param part: Which part(s) of the dataset to prepare ("train", "dev", "test", or "all")
    :return: Dictionary with dataset parts and their recording/supervision sets
    """
    config = DATASET_CONFIGS.get(dataset_name)
    if not config:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    parts_to_process = _validate_parts(part, config["valid_parts"])

    result = {}
    for p in parts_to_process:
        logging.info(f"Preparing {p} part of the {dataset_name} dataset")

        if not (corpus_dir / p).exists():
            logging.warning(
                f"Part {p} does not exist in {corpus_dir}, skipping preparation."
            )
            continue

        metadata = _load_metadata(corpus_dir=corpus_dir, part=p)

        recordings, supervisions = _prepare(
            metadata=metadata,
            text_key=config["text_key"],
            audio_file_path_key="audio_file",
            audio_id_key="id",
            gender_key="gender",
            speaker_key="speaker_id",
            custom=config.get("custom", {}),
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


def prepare_dataset(
    dataset_name: Literal[
        "samromur", "malromur", "althingi", "samromur_queries", "samromur_children"
    ] = "samromur",
    corpus_dir: Pathlike = "data",
    manifest_dir: Optional[Pathlike] = "data/manifests",
    part: Literal["train", "dev", "test", "all"] = "all",
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Prepare manifests for the Samromur, Malromur, Althingi, Samromur Queries, or Samromur Children dataset.

    :param dataset_name: Which dataset to prepare
    :param corpus_dir: Directory where the dataset has been downloaded
    :param manifest_dir: Optional directory where manifests will be stored
    :param part: Which part(s) of the dataset to prepare ("train", "dev", "test", or "all")
    :return: Dictionary with dataset parts and their recording/supervision sets
    """
    manifest_dir = Path(manifest_dir) if manifest_dir else None
    corpus_dir = Path(corpus_dir) / dataset_name
    if manifest_dir is not None:
        manifest_dir.mkdir(parents=True, exist_ok=True)

    return _prepare_dataset(
        corpus_dir=corpus_dir,
        manifest_dir=manifest_dir,
        dataset_name=dataset_name,
        part=part,
    )


# if __name__ == "__main__":
#     # First download the dataset
# download_dataset(
#     target_dir=Path("/home/dem/projects/k2/data"),
#     dataset_name="samromur",
#     part="dev",
# )

# Then prepare the manifests
# prepare_dataset(
#     corpus_dir=Path("/home/dem/projects/k2/data"),
#     part="dev",
#     dataset_name="samromur",
#     manifest_dir=Path("/home/dem/projects/k2/data/manifests"),
# )
