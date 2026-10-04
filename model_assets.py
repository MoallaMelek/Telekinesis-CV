"""Verified local model assets; downloads never touch webcam data."""
from concurrent.futures import CancelledError
import hashlib
from pathlib import Path
import tempfile
import time
from urllib.request import urlopen


def check_cancelled(cancel):
    if cancel is not None and cancel.is_set():
        raise CancelledError("Model loading cancelled")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_asset(path, url, digest, *, cancel=None, max_bytes=64 * 1024 * 1024):
    """Verify cache or atomically install a bounded, checksum-verified download.

    Unique temporary files permit concurrent app starts. Socket and total timeouts
    bound stalls; a bad response never replaces the cache.
    """
    path = Path(path)
    check_cancelled(cancel)
    if path.is_file() and sha256(path) == digest:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        print(f"Downloading {path.name}...", flush=True)
        started = time.monotonic()
        with urlopen(url, timeout=10) as response:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".",
                                             suffix=".download", delete=False) as output:
                temporary = Path(output.name)
                downloaded = 0
                actual = hashlib.sha256()
                while True:
                    check_cancelled(cancel)
                    if time.monotonic() - started > 180:
                        raise TimeoutError("Download exceeded 180 seconds")
                    chunk = response.read1(64 * 1024)
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    if downloaded > max_bytes:
                        raise RuntimeError("Download exceeds the model size limit")
                    actual.update(chunk)
                    output.write(chunk)
        check_cancelled(cancel)
        if actual.hexdigest() != digest:
            raise RuntimeError("Model checksum failed")
        temporary.replace(path)
        return path
    except (OSError, RuntimeError) as error:
        raise RuntimeError(
            f"Model {path.name} could not be downloaded or verified: {error}. "
            f"Connect to the internet once, or place the verified file in {path.parent}.") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
