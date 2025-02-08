import flair

from flair.datasets import NER_GERMAN_GERMEVAL
from flair.datasets.sequence_labeling import ColumnCorpus
from flair.file_utils import cached_path

from pathlib import Path
from typing import Optional, Union


class NER_GERMEVAL_2014_NO_WIKIPEDIA(ColumnCorpus):
    def __init__(
            self,
            base_path: Optional[Union[str, Path]] = None,
            in_memory: bool = True,
            **corpusargs,
    ) -> None:
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)
        dataset_name = self.__class__.__name__.lower()
        data_folder = base_path / dataset_name
        data_path = flair.cache_root / "datasets" / dataset_name

        column_format = {1: "text", 2: "ner"}

        hf_download_path = "https://huggingface.co/datasets/stefan-it/germeval14_no_wikipedia/resolve/main"

        for split in ["train", "dev", "test"]:
            cached_path(f"{hf_download_path}/NER-de-without-wikipedia-{split}.tsv", data_path)

        super().__init__(
            data_folder,
            column_format=column_format,
            column_delimiter="\t",
            document_separator_token=None,
            in_memory=in_memory,
            comment_symbol="#\t",
            **corpusargs,
        )
