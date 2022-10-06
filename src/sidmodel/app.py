"""Command line application entry point.

"""
__author__ = 'Paul Landes'

from typing import Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
import sys
import random
from pathlib import Path
import shutil
import re
from io import StringIO
from zensols.persist import dealloc, PersistedWork
from zensols.config import ConfigFactory
from zensols.cli import ActionCliManager
from zensols.nlp import FeatureDocumentParser
from zensols.dataset import StratifiedStashSplitKeyContainer
from zensols.deeplearn.model import ModelFacade
from zensols.deeplearn.cli import FacadeApplication
from . import (
    MimicNotesExtractor, ResultAnalyzer, NoteFeatureDocument, EmbeddingAnalyzer
)

logger = logging.getLogger(__name__)


class PredictionOutputType(Enum):
    text = auto()
    json = auto()


@dataclass
class Cleaner(object):
    """Remove corpus notes and (somewhat) temporary files.

    """
    clean_dirs: Dict[str, Path] = field()
    """Directories to remove files."""

    def __call__(self):
        for name, path in self.clean_dirs.items():
            if path.is_dir():
                logger.info(f'removing {name} contents at: {path}')
                shutil.rmtree(path)
            else:
                logger.info(f'{name} ({path}) does not exist--skipping')


@dataclass
class Application(FacadeApplication):
    """This contains a basic baseline model for identifying section in clinical
    text.

    """
    CLI_META = ActionCliManager.combine_meta(
        FacadeApplication,
        {'option_includes': set(
            'run input_path model_path limit out_type output_csv output_yml output_file'.split()),
         'mnemonic_overrides': {'extract_mimic_notes': 'extract',
                                'predict_secs': 'predsecs',
                                'dump_hyperparams': 'hyperparams'},
         'option_overrides': {'input_path': {'long_name': 'input'}}})

    config_factory: ConfigFactory = field(default=None)
    cleaner: Cleaner = field(default=None)
    mimic_notes_extractor: MimicNotesExtractor = field(default=None)
    result_analyzer: ResultAnalyzer = field(default=None)
    emb_analyzer: EmbeddingAnalyzer = field(default=None)
    hyperparam_desc_file: Path = field(default=None)

    def __post_init__(self):
        super().__post_init__()
        self._first_batch_pw = PersistedWork(
            '_first_batch_pw', self, cache_global=True)

    def clean(self):
        """Remove corpus notes and (somewhat) temporary files."""
        self.cleaner()

    def _get_doc_parser(self) -> FeatureDocumentParser:
        doc_stash = self.config_factory('note_feature_doc_factory_stash')
        return doc_stash.doc_parser

    def extract_mimic_notes(self):
        """Extract corresponding MIMIC notes and write them to the corpus directory.

        """
        self.mimic_notes_extractor()

    def split(self):
        """Write the stratified splits.

        """
        facade: ModelFacade = self.get_cached_facade()
        stash: StratifiedStashSplitKeyContainer = facade.batch_stash.\
            split_stash_container.split_container
        stash.stratified_write = True
        stash.write()

    def stats(self):
        """Dump annotation statistics."""
        logger.info('dumping stats')
        self.result_analyzer()

    def dump_metrics(self):
        """Dump summary and per label metrics to a LaTeX mktable YAML and CSV files.

        """
        with dealloc(self.create_facade()) as facade:
            self._enable_cli_logging(facade)
            dumper = self.config_factory('perf_metrics_dumper', facade=facade)
            dumper()

    def _pred_secs(self, model_path: Path, input_path: Path):
        with open(input_path) as f:
            sents = [f.read()]
        self.model_path = Path('results/tmpmodel')
        with dealloc(self.create_facade()) as facade:
            docs: Tuple[NoteFeatureDocument] = facade.predict(sents)
            for doc in docs:
                doc.write_headers()

    def predict_secs(self, input_path: Path = Path('target/topred'),
                     model_path: Path = None,
                     out_type: PredictionOutputType = PredictionOutputType.text,
                     limit: int = None):
        """Predict the section IDs of a medical note.

        :param input_path: the path to the medical note to annotate

        :param model_path: the path to the model or use the last trained model
                           if not provided

        :param limit: the max number of document to predict when the input path
                      is a directory

        """
        limit = sys.maxsize if limit is None else limit
        dir_output: Path = None
        if input_path.is_dir():
            paths = list(input_path.iterdir())
            random.shuffle(paths)
            paths = paths[:limit]
            dir_output = Path('target/preds')
            dir_output.mkdir(parents=True, exist_ok=True)
        else:
            paths = [input_path]
        sents = []
        for path in paths:
            with open(path) as f:
                sents.append(f.read())
        self.model_path = model_path
        ext = 'txt' if out_type == PredictionOutputType.text else 'json'
        with dealloc(self.create_facade()) as facade:
            docs: Tuple[NoteFeatureDocument] = facade.predict(sents)
            for path, doc in zip(paths, docs):
                m = re.match(r'^(\d+).txt$', path.name)
                if m is not None:
                    doc.anon_note.row_id = int(m.group(1))
                    path = path.parent / f'{path.stem}.{ext}'
                sio = StringIO()
                if out_type == PredictionOutputType.text:
                    doc.write_headers(writer=sio)
                else:
                    doc.anon_note.asjson(writer=sio, indent=4)
                if dir_output is None:
                    print(sio.getvalue())
                else:
                    fpath = dir_output / f'{path.name}'
                    with open(fpath, 'w') as f:
                        f.write(sio.getvalue())
                    logger.info(f'wrote: {fpath}')

    def dump_hyperparams(self, output_file: Path = Path('hyperparameters.csv')):
        """Write hyperparameters and their descriptions to a CSV file.

        :param output_file: the CSV file to write the hyperparameter data

        """
        import pandas as pd
        fac: ConfigFactory = self.config_factory
        df: pd.DataFrame = pd.read_csv(self.hyperparam_desc_file)
        rows: List[str, str, str, Any] = []
        for section, name, desc in df.itertuples(index=None):
            inst = fac.instance(section)
            val: Any = getattr(inst, name)
            val: str = str(val)
            rows.append((section, name, desc, val))
        df = pd.DataFrame(rows, columns='section name description value'.split())
        df.to_csv(output_file)
        logger.info(f'wrote {output_file}')
