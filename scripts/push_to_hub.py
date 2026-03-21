import argparse
import logging

from lana_radgen.hub import push_directory_to_hub
from lana_radgen.logging_utils import configure_logging

LOGGER = logging.getLogger("push_to_hub")


def main() -> None:
    parser = argparse.ArgumentParser(description="Push a saved model directory to Hugging Face Hub.")
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--commit-message", default="Upload LANA model")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)
    repo_url = push_directory_to_hub(args.local_dir, args.repo_id, args.commit_message)
    LOGGER.info("Uploaded model to %s", repo_url)


if __name__ == "__main__":
    main()
