import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "model" / "model_manifest.json"
DEFAULT_REPOSITORY = "Nikhitha-coder19/RiskPulse"
DEFAULT_RELEASE_TAG = "v1.0.0"
CHUNK_SIZE = 1024 * 1024


def load_manifest():
    try:
        with MANIFEST_PATH.open("r", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Unable to read model manifest: {error}") from error

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise SystemExit("Model manifest must contain a non-empty artifacts list.")

    return manifest, artifacts


def sha256_and_size(path):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(CHUNK_SIZE), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest().upper(), size


def verify_artifact(path, artifact):
    if not path.is_file():
        return False

    actual_hash, actual_size = sha256_and_size(path)
    return (
        actual_hash == artifact["sha256"].upper()
        and actual_size == artifact["size"]
    )


def download_artifact(url, destination, artifact):
    temporary_path = None
    try:
        request = Request(url, headers={"User-Agent": "RiskPulse-model-provisioner"})
        with urlopen(request, timeout=60) as response:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".download",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                for chunk in iter(lambda: response.read(CHUNK_SIZE), b""):
                    temporary_file.write(chunk)

        if not verify_artifact(temporary_path, artifact):
            actual_hash, actual_size = sha256_and_size(temporary_path)
            raise RuntimeError(
                f"Checksum or size mismatch for {artifact['filename']}: "
                f"expected {artifact['sha256']} / {artifact['size']} bytes, "
                f"received {actual_hash} / {actual_size} bytes"
            )

        os.replace(temporary_path, destination)
        temporary_path = None
    except (HTTPError, URLError, OSError, RuntimeError) as error:
        raise SystemExit(f"Failed to provision {artifact['filename']}: {error}") from error
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def provision_models(repository, release_tag):
    manifest, artifacts = load_manifest()
    model_dir = PROJECT_ROOT / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    repository = repository or manifest.get("repository", DEFAULT_REPOSITORY)
    release_tag = release_tag or manifest.get("release_tag", DEFAULT_RELEASE_TAG)

    for artifact in artifacts:
        filename = artifact["filename"]
        asset_name = artifact["asset_name"]
        destination = model_dir / filename
        if verify_artifact(destination, artifact):
            print(f" verified {filename}; download skipped")
            continue

        url = (
            f"https://github.com/{repository}/releases/download/"
            f"{release_tag}/{asset_name}"
        )
        print(f" downloading {asset_name} from release {release_tag}")
        download_artifact(url, destination, artifact)
        print(f" provisioned and verified {filename}")


def main():
    manifest, _ = load_manifest()
    parser = argparse.ArgumentParser(
        description="Provision and verify RiskPulse frozen runtime model artifacts."
    )
    parser.add_argument(
        "--repository",
        default=os.environ.get("RISKPULSE_RELEASE_REPOSITORY", manifest.get("repository")),
        help="GitHub repository in OWNER/REPOSITORY form.",
    )
    parser.add_argument(
        "--tag",
        default=os.environ.get("RISKPULSE_RELEASE_TAG", manifest.get("release_tag")),
        help="GitHub Release tag containing the frozen model assets.",
    )
    args = parser.parse_args()

    if not args.repository or not args.tag:
        raise SystemExit("Both a release repository and release tag are required.")

    provision_models(args.repository, args.tag)
    print("RiskPulse model provisioning complete.")


if __name__ == "__main__":
    main()
