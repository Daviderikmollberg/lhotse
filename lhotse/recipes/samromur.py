import csv
import logging
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Union, List
from tqdm import tqdm
import glob
import subprocess
from lhotse.audio import Recording, RecordingSet
from lhotse.qa import fix_manifests, validate_recordings_and_supervisions
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, resumable_download, safe_extract
from lhotse.kaldi import load_kaldi_data_dir, load_kaldi_text_mapping

DATASET_AUDIO_URLS = {
    "samromur": {
        "dev": "https://huggingface.co/datasets/DavidErikMollberg/samromur_asr/resolve/main/data/dev/audio.tar.gz",
        "test": "https://huggingface.co/datasets/DavidErikMollberg/samromur_asr/resolve/main/data/test/audio.tar.gz",
        "train": "https://huggingface.co/datasets/DavidErikMollberg/samromur_asr/resolve/main/data/train/audio.tar.gz",
    },
    "malromur": {
        "train": "https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/202/malromur.zip"
    },
    "althingi": {
        "base": "https://huggingface.co/datasets/language-and-voice-lab/althingi_asr/resolve/main",
        "repo_base": "https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/277",
        "original_text": "{repo_base}/althingi_texti.tar.gz",
        "train_parts": "{base}/corpus/speech/train/train_part_{part_num}.tar.gz",
        "default": "{base}/corpus/speech/{part}.tar.gz",
        "metadata": "{base}/corpus/files/metadata_{part}.tsv",
    },
    "samromur_queries": "https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/180/samromur_queries_21.12.zip",
    "samromur_children": {
        "dev": "https://huggingface.co/datasets/language-and-voice-lab/samromur_children/resolve/main/corpus/speech/dev.tar.gz",
        "test": "https://huggingface.co/datasets/language-and-voice-lab/samromur_children/resolve/main/corpus/speech/test.tar.gz",
        "train": [
            "https://huggingface.co/datasets/language-and-voice-lab/samromur_children/resolve/main/corpus/speech/train/train_part_01.tar.gz",
            "https://huggingface.co/datasets/language-and-voice-lab/samromur_children/resolve/main/corpus/speech/train/train_part_02.tar.gz",
            "https://huggingface.co/datasets/language-and-voice-lab/samromur_children/resolve/main/corpus/speech/train/train_part_03.tar.gz",
        ],
    },
    "raddromur": {
        "train": "https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/286/RADDROMUR_22.09.zip"
    },
    "ruv_tv": {
        "train": "https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/93/ruv_tv.tar.gz"
    },
    "ruv_tv_unk_speakers": {
        "train": [
            "https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/191/Frettirkl1900_parta.zip",
            "https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/191/Frettirkl1900_partb.zip",
            "https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/191/Frettirkl1900_partc.zip",
            "https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/191/Frettirkl1900_partd.zip",
            "https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/191/StundinOkkar.zip",
            "https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/191/Menningin.zip",
            "https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/191/Krakkafrettir.zip",
            "https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/191/Kiljan.zip",
            "https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/191/Kastljos.zip",
        ],
        "text": "https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/191/text",
    },
    "ismus": {
        "train": "https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/315/audio.zip",
    },
    "icelandic_broadcast_speech": {
        "train": "https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/193",
    },
    "samromur_l2": {
        "train": "https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/263/samromur_L2_22.09.zip",  # There is one archive for all sets
    },
}


METADATA_URL_BASE = "https://huggingface.co/datasets/DavidErikMollberg/asr_metadata/resolve/main/{dataset}_metadata_{part}.csv"

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
    "raddromur": {
        "valid_parts": [
            "train",
        ],
        "text_key": "proper_nouns_cased",
        "custom": {
            "text_normalized": "text_normalized",
            "podcast_id": "podcast_id",
        },
    },
    "ismus": {
        "valid_parts": ["train", "test"],
        "text_key": "proper_nouns_cased",
        "custom": {},
    },
    "ruv_tv_unk_speakers": {"valid_parts": ["train"], "text_key": "text", "custom": {}},
    "icelandic_broadcast_speech": {
        "valid_parts": ["train"],
        "text_key": "text",
        "custom": {
            "speaker_name": "speaker_name",
            "gender": "gender",
            "show": "show",
        },
    },
    "samromur_l2": {
        "valid_parts": ["train", "dev", "test", "all"],
        "text_key": "proper_nouns_cased",
        "custom": {
            "age": "age_range",
            "text_original": "text",
            "text_normalized": "text_normalized",
            "gender": "gender",
            "native_language": "native_language",
            "is_validated": "is_validated",
        },
    },
    "ruv_tv": {"valid_parts": ["train"], "text_key": "text", "custom": {}},
}


def _validate_parts(part: str, dataset_name: List[str]) -> List[str]:
    """
    Validate the part against the list of valid parts.
    Raises ValueError if the part is not valid.
    """
    config = DATASET_CONFIGS.get(dataset_name)
    if not config:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    valid_parts = config["valid_parts"]
    if not isinstance(part, str):
        raise ValueError(f"Part must be a string, got {type(part)} instead.")

    if part not in valid_parts:
        raise ValueError(f"Invalid part: {part}. Choose from {valid_parts}.")

    if part == "all":
        return [p for p in valid_parts if p != "all"]
    else:
        return [part]


def is_empty(path):
    """
    Check the size of a file and whether it is empty.

    Args:
        path (str or Path): Path to the file.

    Returns:
        tuple: (is_empty: bool)
    """
    p = Path(path)
    size = p.stat().st_size
    return size == 0


def _download_metadata(
    part: str, dataset_name: str, part_dir: Path, force_download: bool
) -> Path:
    """
    Download metadata for a dataset part.

    :param part: Which part of the dataset
    :param dataset_name: Name of the dataset
    :param part_dir: Directory where metadata will be saved
    :param force_download: Whether to force download even if file exists
    :return: Path to the downloaded metadata file
    """
    metadata_url = METADATA_URL_BASE.format(part=part, dataset=dataset_name)
    metadata_path = part_dir / "metadata.csv"

    part_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Downloading {part} metadata for {dataset_name} to {metadata_path}")
    resumable_download(
        metadata_url,
        filename=metadata_path,
        force_download=force_download,
    )
    return metadata_path


def _ensure_download_dir(
    dataset_name: str,
    part: str,
    target_dir: Path,
    split: Literal["dev", "test", "train"] = "",
) -> Tuple[Path, Path, Path]:
    """
    Create and return dataset directories.

    :param dataset_name: Name of the dataset
    :param part: Which part of the dataset
    :param target_dir: Base directory for downloads
    :param split: Optionally create a complete detector for each split
    :return: Tuple of (dataset_dir, part_dir, completed_detector)
    """
    dataset_dir = target_dir / dataset_name
    part_dir = dataset_dir / part
    completed_detector = (
        part_dir / f".{split}_download_completed" if split else ".download_completed"
    )

    dataset_dir.mkdir(parents=True, exist_ok=True)
    part_dir.mkdir(parents=True, exist_ok=True)

    return dataset_dir, part_dir, completed_detector


def _resumable_download(
    url: str,
    filename: Path,
    force_download: bool,
):
    """
    Wrapper for resumable_download with additional logging.

    Downloads a file from a URL to a target path with the ability to resume
    interrupted downloads. Skips the download if the file already exists
    and force_download is False.

    :param url: URL of the file to download
    :param filename: Path where the file will be saved
    :param force_download: If True, download even if the file exists
    """

    if filename.exists() and not force_download and not is_empty(filename):
        logging.info(
            f"File {filename} already exists and force_download is False. Skipping download."
        )
        return
    logging.info(f"Starting download from {url} to {filename}")
    resumable_download(
        url,
        filename=filename,
        force_download=force_download,
    )
    logging.info(f"Downloaded {url} to {filename}")


def _download_archive(
    url: str,
    target_path: Path,
    force_download: bool,
    extract_to: Optional[Path] = None,
    extract_format: Optional[str] = None,
    remove_after: bool = False,
) -> None:
    """
    Download and optionally extract an archive file.

    :param url: URL to download from
    :param target_path: Where to save the downloaded file
    :param force_download: Whether to force download even if file exists
    :param extract_to: Directory to extract contents to (if None, don't extract)
    :param extract_format: Format of archive ('tar', 'zip', or None to auto-detect)
    :param remove_after: Whether to remove archive after extraction
    """
    logging.info(f"Downloading archive from {url} to {target_path}")
    _resumable_download(
        url,
        filename=target_path,
        force_download=force_download,
    )

    if extract_to:
        logging.info(f"Extracting archive to {extract_to}")
        if extract_format == "tar" or str(target_path).endswith((".tar.gz", ".tgz")):
            with tarfile.open(target_path) as tar:
                safe_extract(tar, path=extract_to)
        elif extract_format == "zip" or str(target_path).endswith(".zip"):
            with zipfile.ZipFile(target_path, "r") as zip_ref:
                zip_ref.extractall(extract_to)

        if remove_after:
            logging.info(f"Removing archive {target_path}")
            target_path.unlink(missing_ok=True)


def _extract_files_from_zip_by_pattern(
    archive_path: Path,
    output_dir: Path,
    pattern: str,
    desc: str = "Extracting files",
) -> List[Path]:
    """
    Extract files from a ZIP archive that match a specific pattern.

    :param archive_path: Path to the ZIP archive
    :param output_dir: Directory where the extracted files will be saved
    :param pattern: Pattern to match files in the archive (using startswith)
    :param desc: Description for the progress bar
    :return: List of paths to the extracted files
    """
    extracted_files = []

    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        # Get all files in the archive that match the pattern
        part_files = [f for f in zip_ref.namelist() if f.startswith(pattern)]

        if not part_files:
            logging.warning(f"No files found for pattern '{pattern}' in the archive")
        for file in tqdm(part_files, desc=desc):
            filename = Path(file).name
            file_content = zip_ref.read(file)
            output_path = output_dir / filename
            with open(output_path, "wb") as f_out:
                f_out.write(file_content)
            extracted_files.append(output_path)

    return extracted_files


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
        audio_url = DATASET_AUDIO_URLS[dataset_name][p]

        logging.info(f"Downloading {p} audio to {audio_tar_path}")
        _download_archive(
            audio_url,
            filename=audio_tar_path,
            force_download=force_download,
        )

        metadata_url = METADATA_URL_BASE.format(part=p, dataset=dataset_name)
        metadata_path = part_dir / "metadata.csv"
        if not part_dir.exists():
            part_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Downloading {p} metadata to {metadata_path}")
        _download_archive(
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
    parts = _validate_parts(part, dataset_name)

    target_dir = Path(target_dir)
    dataset_dir = target_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    for p in tqdm(parts, desc="Downloading Samromur Children parts"):
        _, part_dir, completed_detector = _ensure_download_dir(
            dataset_name, p, target_dir
        )

        if completed_detector.is_file() and not force_download:
            logging.info(
                f"Skipping {p} part of {dataset_name} because {completed_detector} exists."
            )
            downloaded_parts[p] = part_dir
            continue

        audio_dir = part_dir / "audio"
        audio_dir.mkdir(exist_ok=True)

        # Download metadata
        _download_metadata(p, dataset_name, part_dir, force_download)

        if p == "train":
            for idx, url in enumerate(DATASET_AUDIO_URLS[dataset_name]["train"], 1):
                part_num = f"{idx:02d}"
                audio_tar_path = part_dir / f"train_part_{part_num}.tar.gz"

                # Download and extract audio archive
                _download_archive(
                    url,
                    audio_tar_path,
                    force_download,
                    extract_to=part_dir,
                    remove_after=remove_archive,
                )

            extracted_files = list(part_dir.glob("**/*.flac"))
            for flac_file in tqdm(extracted_files, desc=f"Moving {p} audio files"):
                shutil.move(str(flac_file), str(audio_dir / flac_file.name))

            # Clean up extracted directories
            for idx in range(1, 4):
                part_num = f"{idx:02d}"
                shutil.rmtree(part_dir / f"train_part_{part_num}", ignore_errors=True)
        else:
            audio_tar_path = part_dir / f"{p}.tar.gz"

            # Download and extract audio archive
            _download_archive(
                DATASET_AUDIO_URLS[dataset_name][p],
                audio_tar_path,
                force_download,
                extract_to=part_dir,
                remove_after=remove_archive,
            )

            extracted_files = list(part_dir.glob("**/*.flac"))
            for flac_file in tqdm(extracted_files, desc=f"Moving {p} audio files"):
                shutil.move(str(flac_file), str(audio_dir / flac_file.name))

            # Clean up extracted directory
            shutil.rmtree(part_dir / p, ignore_errors=True)

        logging.info(f"Processed and moved audio files for {p} part of {dataset_name}")

        completed_detector.touch()
        downloaded_parts[p] = part_dir
        logging.info(f"Downloaded and prepared {p} part of {dataset_name}")

    return downloaded_parts


def _download_icelandic_broadcast_speech(
    part: str, target_dir: Pathlike, force_download: bool, remove_archive: bool
) -> Dict[str, Path]:
    """
    Download the Icelandic Broadcast Speech dataset.

    :param part: Which part(s) of the dataset to download ("train" or "all")
    :param target_dir: Directory where the dataset will be downloaded
    :param force_download: If True, force redownload even if files exist
    :param remove_archive: If True, remove the downloaded archive files after extraction
    :return: Dictionary mapping parts to their directory paths
    """
    downloaded_parts = {}
    dataset_name = "icelandic_broadcast_speech"
    parts = _validate_parts(part, dataset_name)
    p = parts[0]
    dataset_dir, part_dir, completed_detector = _ensure_download_dir(
        dataset_name, p, Path(target_dir)
    )

    if completed_detector.is_file() and not force_download:
        logging.info(
            f"Skipping {p} part of {dataset_name} because {completed_detector} exists."
        )
        downloaded_parts[p] = part_dir
        return downloaded_parts

    base_url = DATASET_AUDIO_URLS[dataset_name]["train"]

    # Download split zip files and main zip
    for i in range(1, 7):
        url = base_url + f"/cut_audio_RELEASE.z0{i}"
        target_path = part_dir / f"cut_audio_RELEASE.z0{i}"
        _resumable_download(url, filename=target_path, force_download=force_download)
    url = base_url + "/cut_audio_RELEASE.zip"
    target_path = part_dir / "cut_audio_RELEASE.zip"
    _resumable_download(url, filename=target_path, force_download=force_download)

    # Download metadata.csv from METADATA_URL_BASE
    metadata_url = METADATA_URL_BASE.format(part=p, dataset=dataset_name)
    metadata_path = part_dir / "metadata.csv"
    logging.info(f"Downloading {p} metadata to {metadata_path}")
    _resumable_download(
        metadata_url,
        filename=metadata_path,
        force_download=force_download,
    )

    # Combine the split zip files
    logging.info("Combining split zip files...")
    combined_zip = part_dir / "unsplit.zip"
    if not combined_zip.exists():
        subprocess.run(
            [
                "zip",
                "-s",
                "0",
                str(part_dir / "cut_audio_RELEASE.zip"),
                "--out",
                str(combined_zip),
            ],
            check=True,
        )
    # Extract the combined zip file
    logging.info(f"Extracting audio files to {part_dir}")

    with zipfile.ZipFile(combined_zip, "r") as zip_ref:
        zip_ref.extractall(part_dir)

    shutil.move(part_dir / "audio_RELEASE", part_dir / "audio")

    # Prepare the audio direc
    completed_detector.touch()

    if remove_archive:
        for i in range(1, 7):
            (part_dir / f"cut_audio_RELEASE.z0{i}").unlink(missing_ok=True)
        (part_dir / "cut_audio_RELEASE.zip").unlink(missing_ok=True)
        combined_zip.unlink(missing_ok=True)

    downloaded_parts[p] = part_dir
    logging.info(f"Downloaded and prepared {p} part of {dataset_name}")

    return downloaded_parts


def _download_malromur(
    part: str, target_dir: Pathlike, force_download: bool, remove_archive: bool
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
        DATASET_AUDIO_URLS[dataset_name][part],
        filename=malromur_zip_path,
        force_download=force_download,
    )
    with zipfile.ZipFile(malromur_zip_path, "r") as zip_ref:
        all_files = zip_ref.namelist()
        correct_files = [
            f for f in all_files if "correct/" in f and not f.endswith("/")
        ]
        for file in tqdm(correct_files, desc="Extracting correct audio files"):
            filename = Path(file).name
            file_content = zip_ref.read(file)
            with open(audio_dir / filename, "wb") as f_out:
                f_out.write(file_content)

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

    parts = _validate_parts(part, dataset_name)

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
        original_text_url = DATASET_AUDIO_URLS[dataset_name]["original_text"].format(
            repo_base=DATASET_AUDIO_URLS[dataset_name]["repo_base"]
        )

        _download_archive(
            original_text_url,
            original_text_archive,
            force_download,
            extract_to=althingi_dir,
            remove_after=remove_archive,
        )

        original_text_completed.touch()

    for p in tqdm(parts, desc="Downloading Althingi parts"):
        _, part_dir, completed_detector = _ensure_download_dir(
            dataset_name, p, target_dir
        )

        if completed_detector.is_file() and not force_download:
            logging.info(
                f"Skipping {p} part of {dataset_name} because {completed_detector} exists."
            )
            downloaded_parts[p] = part_dir
            continue

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
        metadata_url = DATASET_AUDIO_URLS[dataset_name]["metadata"].format(
            base=DATASET_AUDIO_URLS[dataset_name]["base"], part=p
        )
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
                audio_url = DATASET_AUDIO_URLS[dataset_name]["train_parts"].format(
                    base=DATASET_AUDIO_URLS[dataset_name]["base"], part_num=part_num
                )

                # Download and extract audio archive
                _download_archive(
                    audio_url,
                    audio_tar_path,
                    force_download,
                    extract_to=part_dir,
                    remove_after=remove_archive,
                )

            flac_files = glob.glob(
                str(part_dir / f"train_part_*" / "*" / "*" / "*.flac")
            )
        else:
            audio_tar_path = part_dir / f"{p}.tar.gz"
            audio_url = DATASET_AUDIO_URLS[dataset_name]["default"].format(
                base=DATASET_AUDIO_URLS[dataset_name]["base"], part=p
            )

            # Download and extract audio archive
            _download_archive(
                audio_url,
                audio_tar_path,
                force_download,
                extract_to=part_dir,
                remove_after=remove_archive,
            )

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
                f"audio/{original_texts[audio_id_without_spk]['utterance_id']}.flac"
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
    parts = _validate_parts(part, dataset_name)

    target_dir = Path(target_dir)
    dataset_dir = target_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    archive_path = dataset_dir / "samromur_queries_21.12.zip"
    # Create a special marker for the entire archive extraction
    archive_extracted_marker = dataset_dir / ".archive_extracted"

    # Check if we need to download and extract the common archive
    need_extraction = force_download or not archive_extracted_marker.is_file()

    if need_extraction:
        # Download the archive
        _download_archive(
            DATASET_AUDIO_URLS[dataset_name],
            archive_path,
            force_download,
            remove_after=remove_archive,
        )

        logging.info(f"Extracting Samromur Queries dataset to {dataset_dir}")
        full_metadata_path = dataset_dir / "metadata.tsv"
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            file_content = zip_ref.read("samromur_queries_21.12/metadata.tsv")
            with open(full_metadata_path, "wb") as f_out:
                f_out.write(file_content)

            for p in parts:
                _, part_dir, _ = _ensure_download_dir(dataset_name, p, target_dir)
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

        # Mark the archive as extracted
        archive_extracted_marker.touch()
    else:
        logging.info(
            f"Skipping archive extraction as {archive_extracted_marker} exists."
        )

    # Process each part
    for p in parts:
        _, part_dir, completed_detector = _ensure_download_dir(
            dataset_name, p, target_dir
        )

        # Skip this part if already completed
        if completed_detector.is_file() and not force_download:
            logging.info(
                f"Skipping {p} part of {dataset_name} because {completed_detector} exists."
            )
            downloaded_parts[p] = part_dir
            continue

        # Ensure the part directory exists
        part_dir.mkdir(parents=True, exist_ok=True)

        # Ensure audio directory exists
        audio_dir = part_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        # Download metadata for each part
        _download_metadata(p, dataset_name, part_dir, force_download)

        # Mark this part as completed
        completed_detector.touch()
        downloaded_parts[p] = part_dir
        logging.info(f"Prepared {p} part of {dataset_name}")

    return downloaded_parts


def _download_raddromur(
    part: str, target_dir: Pathlike, force_download: bool, remove_archive: bool
) -> Dict[str, Path]:
    """
    Download the Raddrómur dataset from the specified URL.

    The Raddrómur Corpus is an Icelandic corpus created by the Language and Voice Laboratory (LVL) at
    Reykjavík University (RU) in 2022, made out of radio podcasts mostly taken from RÚV (ruv.is).

    :param part: Which part of the dataset to download (only "train" is available)
    :param target_dir: Directory where the dataset will be downloaded
    :param force_download: If True, force redownload even if files exist
    :param remove_archive: If True, remove the downloaded files after processing
    :return: Dictionary mapping parts to their directory paths
    """
    downloaded_parts = {}
    dataset_name = "raddromur"
    p = _validate_parts(part, dataset_name)[0]

    target_dir = Path(target_dir)

    _, part_dir, completed_detector = _ensure_download_dir(dataset_name, p, target_dir)

    if completed_detector.is_file() and not force_download:
        logging.info(
            f"Skipping {p} part of {dataset_name} because {completed_detector} exists."
        )
        downloaded_parts[p] = part_dir
        return downloaded_parts

    audio_dir = part_dir / "audio"
    audio_dir.mkdir(exist_ok=True, parents=True)

    # Download metadata
    _ = _download_metadata(p, dataset_name, part_dir, force_download)

    # Download and extract the Raddrómur ZIP archive
    archive_path = part_dir / "RADDROMUR_22.09.zip"

    logging.info(f"Downloading Raddrómur archive to {archive_path}")
    _download_archive(
        DATASET_AUDIO_URLS[dataset_name][p],
        archive_path,
        force_download,
        extract_to=part_dir,
        extract_format="zip",
        remove_after=remove_archive,
    )

    # Move audio files to the audio directory
    extracted_dir = part_dir / "RADDROMUR_22.09"
    if extracted_dir.exists():
        audio_files = list(extracted_dir.glob("speech/**/*.flac"))
        logging.info(f"Found {len(audio_files)} WAV files to process")

        for audio_file in tqdm(audio_files, desc="Moving audio files"):
            destination = audio_dir / audio_file.name
            shutil.copy2(audio_file, destination)

        logging.info(f"Moved {len(audio_files)} audio files to {audio_dir}")
        shutil.rmtree(extracted_dir, ignore_errors=True)
    else:
        logging.warning(f"Extracted directory not found: {extracted_dir}")

    completed_detector.touch()
    downloaded_parts[p] = part_dir
    logging.info(f"Downloaded and prepared {p} part of {dataset_name}")

    return downloaded_parts


def _download_ruv_tv(
    part: str, target_dir: Pathlike, force_download: bool, remove_archive: bool
) -> Dict[str, Path]:
    """
    Download the RÚV TV dataset from the specified URL.

    The RUV TV set is 6 hours and 43 minutes of TV data from RÚV, from two talk shows
    and the news: Kastljós (news commentary), Kiljan (literature discussions), and the
    prime time news (Fréttir kl. 19:00). The data contains 5880 utterances from 151 speakers.

    :param part: Which part of the dataset to download (only "train" is available)
    :param target_dir: Directory where the dataset will be downloaded
    :param force_download: If True, force redownload even if files exist
    :param remove_archive: If True, remove the downloaded files after processing
    :return: Dictionary mapping parts to their directory paths
    """
    downloaded_parts = {}
    dataset_name = "ruv_tv"
    p = _validate_parts(part, dataset_name)[0]

    target_dir = Path(target_dir)

    _, part_dir, completed_detector = _ensure_download_dir(dataset_name, p, target_dir)

    if completed_detector.is_file() and not force_download:
        logging.info(
            f"Skipping {p} part of {dataset_name} because {completed_detector} exists."
        )
        downloaded_parts[p] = part_dir
        return downloaded_parts

    # Download and extract the RÚV TV dataset tar.gz archive
    archive_path = part_dir / "ruv_tv.tar.gz"

    logging.info(f"Downloading RÚV TV dataset to {archive_path}")
    _download_archive(
        DATASET_AUDIO_URLS[dataset_name][p],
        archive_path,
        force_download,
        extract_to=part_dir,
        extract_format="tar",
        remove_after=remove_archive,
    )
    shutil.move(str(part_dir / "ruv_tv" / "audio"), str(part_dir / "audio"))
    shutil.move(str(part_dir / "ruv_tv" / "kaldi_data"), str(part_dir / "kaldi_data"))

    completed_detector.touch()
    downloaded_parts[p] = part_dir
    logging.info(f"Downloaded and prepared {p} part of {dataset_name}")

    return downloaded_parts


def _download_ismus(
    part: str, target_dir: Pathlike, force_download: bool, remove_archive: bool
) -> Dict[str, Path]:
    """
    Download the ISMUS dataset from the specified URL.

    The ISMUS corpus is an ASR corpus for Icelandic oral histories from the Árni Magnússon
    Institute for Icelandic Studies (available on ismus.is). It contains 146 hours of
    transcribed audio from interviews, with a training set (~137 hours) and test set (~9 hours).

    :param part: Which part of the dataset to download ("train", "test", or "all")
    :param target_dir: Directory where the dataset will be downloaded
    :param force_download: If True, force redownload even if files exist
    :param remove_archive: If True, remove the downloaded archive after extraction
    :return: Dictionary mapping parts to their directory paths
    """
    downloaded_parts = {}
    dataset_name = "ismus"
    parts = _validate_parts(part, dataset_name)

    target_dir = Path(target_dir)
    dataset_dir = target_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # This archive contains both train and test data
    archive_path = dataset_dir / "audio.zip"

    # Download the archive (the same for both train and test)
    logging.info(f"Downloading ISMUS dataset to {archive_path}")
    resumable_download(
        DATASET_AUDIO_URLS[dataset_name]["train"],
        filename=archive_path,
        force_download=force_download,
    )

    # Process each requested part
    for p in parts:
        _, part_dir, completed_detector = _ensure_download_dir(
            dataset_name, p, target_dir
        )

        # Skip this part if already completed
        if completed_detector.is_file() and not force_download:
            logging.info(
                f"Skipping {p} part of {dataset_name} because {completed_detector} exists."
            )
            downloaded_parts[p] = part_dir
            continue

        # Ensure directories exist
        audio_dir = part_dir / "audio"
        audio_dir.mkdir(exist_ok=True, parents=True)

        # Extract audio files for this part from the zip archive
        logging.info(f"Extracting {p} audio files from archive")
        pattern = f"audio/{p}_flac/"

        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            # Get all files in the archive that match the pattern for this part
            part_files = [f for f in zip_ref.namelist() if f.startswith(pattern)]

            if not part_files:
                logging.warning(
                    f"No files found for pattern '{pattern}' in the archive"
                )

            for file in tqdm(part_files, desc=f"Extracting {p} audio files"):
                if file.endswith(".flac"):  # Only extract audio files
                    filename = Path(file).name
                    file_content = zip_ref.read(file)
                    output_path = audio_dir / filename
                    with open(output_path, "wb") as f_out:
                        f_out.write(file_content)

            logging.info(f"Extracted {len(part_files)} files for part {p}")

        # Download metadata for each part
        _download_metadata(p, dataset_name, part_dir, force_download)

        # Mark this part as completed
        completed_detector.touch()
        downloaded_parts[p] = part_dir
        logging.info(f"Prepared {p} part of {dataset_name}")

    # Remove archive if requested
    if remove_archive:
        logging.info(f"Removing archive {archive_path}")
        archive_path.unlink(missing_ok=True)

    return downloaded_parts


def _download_ruv_tv_unk_speakers(
    part: str, target_dir: Pathlike, force_download: bool, remove_archive: bool
) -> Dict[str, Path]:
    """
    Download the RÚV TV unknown speakers dataset.

    The RUV TV unknown speakers corpus is 281 hours of TV data from six RÚV TV shows.
    The data contains 221,759 utterances from various unlabelled speakers.
    Audio is stored in 16kHz one channel FLAC files created from the original .mp4 episodes.

    Shows included:
    - Fréttir kl. 19:00 (prime time news)
    - Kastljós (news commentary)
    - Kiljan (literature discussion)
    - Krakkafréttir (news for children)
    - Menningin (arts and culture show)
    - Stundin Okkar (children's variety show)

    :param part: Which part of the dataset to download (only "train" is available)
    :param target_dir: Directory where the dataset will be downloaded
    :param force_download: If True, force redownload even if files exist
    :param remove_archive: If True, remove the downloaded files after processing
    :return: Dictionary mapping parts to their directory paths
    """
    downloaded_parts = {}
    dataset_name = "ruv_tv_unk_speakers"
    p = _validate_parts(part, dataset_name)[0]

    target_dir = Path(target_dir)
    _, part_dir, completed_detector = _ensure_download_dir(dataset_name, p, target_dir)

    if completed_detector.is_file() and not force_download:
        logging.info(
            f"Skipping {p} part of {dataset_name} because {completed_detector} exists."
        )
        downloaded_parts[p] = part_dir
        return downloaded_parts

    # Create audio directory
    audio_dir = part_dir / "audio"
    audio_dir.mkdir(exist_ok=True, parents=True)

    # Download text file
    text_file_path = part_dir / "text"
    logging.info(f"Downloading {dataset_name} text file to {text_file_path}")
    resumable_download(
        DATASET_AUDIO_URLS[dataset_name]["text"],
        filename=text_file_path,
        force_download=force_download,
    )

    # Download and extract all zip archives
    archive_mapping = {}
    for archive_url in tqdm(
        DATASET_AUDIO_URLS[dataset_name]["train"], desc="Downloading archives"
    ):
        archive_name = Path(archive_url).name
        archive_path = part_dir / archive_name

        # Extract show name from archive name
        show_name_full = archive_name.split(".")[0]
        # Handle special case for Frettirkl1900 parts
        if "_part" in show_name_full:
            show_name = show_name_full.split("_")[0]

        # Download archive
        logging.info(f"Downloading {archive_name} to {archive_path}")
        resumable_download(
            archive_url,
            filename=archive_path,
            force_download=force_download,
        )

        # Extract archive
        logging.info(f"Extracting {archive_name} to {audio_dir}")

        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            for file_info in tqdm(
                zip_ref.infolist(), desc=f"Extracting {archive_name}"
            ):
                if file_info.filename.endswith("/"):  # Skip directories
                    continue
                if file_info.filename.endswith(".flac"):
                    path_parts = Path(file_info.filename).parts
                    if len(path_parts) > 1 and path_parts[0] == show_name:
                        path_parts = path_parts[1:]
                    target_path = audio_dir / path_parts[-1]
                    file_content = zip_ref.read(file_info.filename)
                    with open(target_path, "wb") as f_out:
                        f_out.write(file_content)
                    archive_mapping[path_parts[-1].replace(".flac", "")] = (
                        show_name_full
                    )

        if remove_archive:
            logging.info(f"Removing archive {archive_path}")
            archive_path.unlink(missing_ok=True)

    # Create metadata.csv file with information from text file

    logging.info(f"Creating metadata file at {part_dir / 'metadata.csv'}")
    text = load_kaldi_text_mapping(text_file_path)
    metadata_path = part_dir / "metadata.csv"

    # There are some empty files for some reason, we will skip those
    empty_audio_files = [
        Path(f).name.replace(".flac", "")
        for f in glob.glob(str(audio_dir / "*.flac"))
        if is_empty(f)
    ]

    with open(metadata_path, "w", encoding="utf-8") as f_out:
        f_out.write(",".join(["file_name", "id", "show", "text"]) + "\n")
        for id, text in tqdm(text.items(), desc="Writing metadata"):
            if id in empty_audio_files:
                logging.warning(f"Skipping empty audio file for ID: {id}")
                continue
            show = archive_mapping.get(id, "")
            if not show:
                logging.warning(f"No show mapping found for ID: {id}")
                continue
            f_out.write(f"audio/{id}.flac,{id},{show},{text}\n")

    completed_detector.touch()
    downloaded_parts[p] = part_dir
    logging.info(f"Downloaded and prepared {p} part of {dataset_name}")

    return downloaded_parts


def _download_samromur_l2(
    part: str, target_dir: Pathlike, force_download: bool, remove_archive: bool
) -> Dict[str, Path]:
    """
    Download the Samromur L2 dataset (non-native Icelandic speakers).

    :param part: Which part(s) of the dataset to download ("train", "dev", "test", or "all")
    :param target_dir: Directory where the dataset will be downloaded
    :param force_download: If True, force redownload even if files exist
    :param remove_archive: If True, remove the downloaded archive files after extraction
    :return: Dictionary mapping parts to their directory paths
    """
    downloaded_parts = {}
    dataset_name = "samromur_l2"
    parts = _validate_parts(part, dataset_name)

    target_dir = Path(target_dir)

    # Download and extract audio archive
    archive_path = target_dir / dataset_name / f"audio.zip"
    _resumable_download(
        url=DATASET_AUDIO_URLS[dataset_name]["train"],
        filename=archive_path,
        force_download=force_download,
    )

    for p in tqdm(parts, desc="Downloading Samromur L2 parts"):
        dataset_dir, part_dir, completed_detector = _ensure_download_dir(
            dataset_name, p, target_dir, split=p
        )

        if completed_detector.is_file() and not force_download:
            logging.info(
                f"Skipping {p} part of {dataset_name} because {completed_detector} exists."
            )
            downloaded_parts[p] = part_dir
            continue

        audio_dir = part_dir / "audio"
        audio_dir.mkdir(exist_ok=True)

        # Use the new extraction function
        pattern = f"audio/{p}/*/*.flac"
        extracted_files = _extract_files_from_zip_by_pattern(
            archive_path=archive_path,
            output_dir=audio_dir,
            pattern=pattern,
            desc=f"Extracting {p} audio files",
        )

        # Download metadata
        _download_metadata(p, dataset_name, part_dir, force_download)

        logging.info(f"Processed audio files for {p} part of {dataset_name}")
        logging.info(f"Extracted {len(extracted_files)} files for part {p}")

        completed_detector.touch()
        downloaded_parts[p] = part_dir
        logging.info(f"Downloaded and prepared {p} part of {dataset_name}")

    if remove_archive and archive_path.exists():
        logging.info(f"Removing archive {archive_path}")
        archive_path.unlink(missing_ok=True)

    return downloaded_parts


def download_dataset(
    target_dir: Pathlike = ".",
    dataset_name: Literal[
        "samromur",
        "malromur",
        "samromur_queries",
        "samromur_children",
        "samromur_l2",
        "althingi",
        "raddromur",
        "ruv_tv",
        "ismus",
        "ruv_tv_unk_speakers",
        "icelandic_broadcast_speech",
    ] = "samromur",
    force_download: Optional[bool] = False,
    part: Literal["train", "dev", "test", "all"] = "all",
    remove_archive: bool = False,
) -> Dict[str, Path]:
    """
    Download an Icelandic speech dataset.

    Available datasets:
    - samromur: General Icelandic speech dataset collected through a web-based platform
    - malromur: Icelandic speech dataset recorded in various environments
    - althingi: Speech dataset from the Icelandic Parliament (Alþingi)
    - samromur_queries: Icelandic voice query dataset collected through a web interface
    - samromur_children: Icelandic speech dataset specifically targeting children's speech
    - raddromur: Icelandic speech corpus made from radio podcasts, created by LVL at Reykjavík University
    - ruv_tv: RÚV TV dataset with 6+ hours of talk shows and news from Icelandic National Broadcasting Service
    - ismus: ISL speech corpus from ISMUS.is
    - ruv_tv_unk_speakers: RÚV TV dataset with unknown speakers
    - icelandic_broadcast_speech: 193 hours of radio and TV data from RÚV (2020-2021)

    Most datasets are available in train/dev/test splits, except malromur, raddromur, ruv_tv, and
    icelandic_broadcast_speech which are train-only.

    :param target_dir: Directory where the dataset will be downloaded
    :param dataset_name: Which dataset to download
    :param force_download: If True, force redownload even if files exist
    :param part: Which part(s) of the dataset to download ("train", "dev", "test", or "all")
    :param remove_archive: If True, remove the downloaded archives after extraction to save space
    :return: Dictionary mapping parts to their directory paths
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    download_functions = {
        "samromur": lambda: _download_samromur(
            part, target_dir, force_download, remove_archive
        ),
        "malromur": lambda: _download_malromur(
            part, target_dir, force_download, remove_archive
        ),
        "althingi": lambda: _download_althingi(
            part, target_dir, force_download, remove_archive
        ),
        "samromur_queries": lambda: _download_samromur_queries(
            part, target_dir, force_download, remove_archive
        ),
        "samromur_children": lambda: _download_samromur_children(
            part, target_dir, force_download, remove_archive
        ),
        "raddromur": lambda: _download_raddromur(
            part, target_dir, force_download, remove_archive
        ),
        "ruv_tv": lambda: _download_ruv_tv(
            part, target_dir, force_download, remove_archive
        ),
        "ismus": lambda: _download_ismus(
            part, target_dir, force_download, remove_archive
        ),
        "ruv_tv_unk_speakers": lambda: _download_ruv_tv_unk_speakers(
            part, target_dir, force_download, remove_archive
        ),
        "icelandic_broadcast_speech": lambda: _download_icelandic_broadcast_speech(
            part, target_dir, force_download, remove_archive
        ),
        "samromur_l2": lambda: _download_samromur_l2(
            part, target_dir, force_download, remove_archive
        ),
    }

    if dataset_name not in download_functions:
        valid_datasets = ", ".join(download_functions.keys())
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Available options: {valid_datasets}"
        )

    logging.info(f"Downloading part/s {part} of {dataset_name} dataset to {target_dir}")

    return download_functions[dataset_name]()


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

    parts_to_process = _validate_parts(part, dataset_name)

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


def _prepare_ruv_tv(
    corpus_dir: Pathlike,
    manifest_dir: Optional[Pathlike] = None,
    dataset_name: str = "samromur",
    part: Literal["train", "dev", "test", "all"] = "all",
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:

    part_dir = Path(corpus_dir) / part
    kaldi_data = part_dir / "kaldi_data"
    wav_scp = []
    with open(kaldi_data / "wav.scp", "r") as f:
        for line in f:
            wav_scp.append(
                line.strip().replace(
                    "/data/ruv-di/version0002/wav/", str(part_dir / "audio") + "/"
                )
            )
    with open(kaldi_data / "wav.scp", "w") as f:
        f.write("\n".join(wav_scp))

    recording_set, supervision_set, _ = load_kaldi_data_dir(
        kaldi_data, sampling_rate=16000, use_reco2dur=False, num_jobs=10
    )
    recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
    validate_recordings_and_supervisions(recording_set, supervision_set)

    if manifest_dir is not None:
        supervision_set.to_file(
            manifest_dir / f"{dataset_name}_supervisions_{part}.jsonl"
        )
        recording_set.to_file(manifest_dir / f"{dataset_name}_recordings_{part}.jsonl")
    result = {}
    result[part] = {
        "recordings": recording_set,
        "supervisions": supervision_set,
    }
    return result


def prepare_dataset(
    dataset_name: Literal[
        "samromur",
        "malromur",
        "althingi",
        "samromur_queries",
        "samromur_children",
        "raddromur",
        "ruv_tv",
        "ismus",
        "ruv_tv_unk_speakers",
        "samromur_l2",
    ] = "samromur",
    corpus_dir: Pathlike = "data",
    manifest_dir: Optional[Pathlike] = "data/manifests",
    part: Literal["train", "dev", "test", "all"] = "all",
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Prepare manifests for the Samromur, Malromur, Althingi, Samromur Queries, Samromur Children, or Raddrómur dataset.

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

    _ = _validate_parts(part, dataset_name)
    if dataset_name == "ruv_tv":
        return _prepare_ruv_tv(
            corpus_dir=corpus_dir,
            manifest_dir=manifest_dir,
            dataset_name=dataset_name,
            part=part,
        )

    return _prepare_dataset(
        corpus_dir=corpus_dir,
        manifest_dir=manifest_dir,
        dataset_name=dataset_name,
        part=part,
    )


if __name__ == "__main__":
    # First download the dataset
    # download_dataset(
    #     target_dir=Path("/home/dem/projects/k2/data"),
    #     dataset_name="samromur_l2",
    #     part="all",
    # )

    # Then prepare the manifests
    prepare_dataset(
        corpus_dir=Path("/home/dem/projects/k2/data"),
        part="train",
        dataset_name="ruv_tv",
        manifest_dir=Path("/home/dem/projects/k2/data/manifests"),
    )
