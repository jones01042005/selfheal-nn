import hashlib
import logging
import os
import tarfile
import time
import urllib.error
import urllib.request

from torchvision.datasets import CIFAR10
from torchvision.datasets.utils import check_integrity


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("cifar10-preparation")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
ARCHIVE_PATH = os.path.join(DATA_DIR, CIFAR10.filename)
ARCHIVE_MD5 = CIFAR10.tgz_md5
DOWNLOAD_ATTEMPTS = 3
DOWNLOAD_TIMEOUT_SECONDS = 60
CHUNK_SIZE = 1024 * 1024


def _datasets_are_valid() -> bool:
    return all(
        check_integrity(
            os.path.join(DATA_DIR, CIFAR10.base_folder, filename),
            expected_md5,
        )
        for filename, expected_md5 in CIFAR10.train_list + CIFAR10.test_list
    )


def _safe_extract(archive: tarfile.TarFile, destination: str) -> None:
    destination = os.path.realpath(destination)
    members = archive.getmembers()

    for member in members:
        if member.issym() or member.islnk():
            raise ValueError(
                f"Refusing CIFAR-10 archive link entry: {member.name!r}"
            )
        if not (member.isdir() or member.isfile()):
            raise ValueError(
                f"Refusing unsupported CIFAR-10 archive entry: {member.name!r}"
            )

        member_path = os.path.realpath(
            os.path.join(destination, member.name.replace("\\", os.sep))
        )
        try:
            is_within_destination = (
                os.path.commonpath((destination, member_path)) == destination
            )
        except ValueError:
            is_within_destination = False

        if not is_within_destination:
            raise ValueError(
                f"Refusing CIFAR-10 archive path outside destination: "
                f"{member.name!r}"
            )

    archive.extractall(destination, members=members)


def _download_archive() -> None:
    url = CIFAR10.url

    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        temporary_path = f"{ARCHIVE_PATH}.part"
        digest = hashlib.md5()
        downloaded = 0
        log.info(
            "Downloading CIFAR-10 archive (attempt %d/%d)",
            attempt,
            DOWNLOAD_ATTEMPTS,
        )

        try:
            with urllib.request.urlopen(
                url,
                timeout=DOWNLOAD_TIMEOUT_SECONDS,
            ) as response, open(temporary_path, "wb") as output:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break

                    output.write(chunk)
                    digest.update(chunk)
                    downloaded += len(chunk)

                    if downloaded % (10 * CHUNK_SIZE) < len(chunk):
                        log.info(
                            "CIFAR-10 download progress: %.1f MiB",
                            downloaded / (1024 * 1024),
                        )

            if digest.hexdigest() != ARCHIVE_MD5:
                raise ValueError("Downloaded CIFAR-10 archive checksum mismatch.")

            os.replace(temporary_path, ARCHIVE_PATH)
            return
        except (OSError, urllib.error.URLError, ValueError):
            log.exception(
                "CIFAR-10 archive download failed on attempt %d/%d",
                attempt,
                DOWNLOAD_ATTEMPTS,
            )
            if os.path.exists(temporary_path):
                os.remove(temporary_path)

            if attempt == DOWNLOAD_ATTEMPTS:
                raise

            time.sleep(attempt * 5)


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    if _datasets_are_valid():
        log.info("CIFAR-10 train and validation files already verified.")
        return

    if not check_integrity(ARCHIVE_PATH, ARCHIVE_MD5):
        _download_archive()

    log.info("Extracting verified CIFAR-10 archive into %s", DATA_DIR)
    with tarfile.open(ARCHIVE_PATH, "r:gz") as archive:
        _safe_extract(archive, DATA_DIR)

    if not _datasets_are_valid():
        raise RuntimeError(
            "CIFAR-10 extraction completed, but one or more dataset files "
            "failed checksum validation."
        )

    os.remove(ARCHIVE_PATH)
    log.info("CIFAR-10 train and validation files verified.")


if __name__ == "__main__":
    main()
