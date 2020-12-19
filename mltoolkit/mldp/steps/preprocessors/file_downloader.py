from .base_preprocessor import BasePreProcessor
import urllib
import os


class FileDownloader(BasePreProcessor):
    """A simple file downloader preprocessing step."""

    def __call__(self, remote_url, local_path):
        if not os.path.exists(local_path):
            urllib.urlretrieve(remote_url, local_path)
        return {"data_path": local_path}
