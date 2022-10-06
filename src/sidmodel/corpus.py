"""Annotation classes and code to read and process them.

"""
__author__ = 'Paul Landes'

from typing import Dict, Any, Tuple, Iterable, Type, List
from dataclasses import dataclass, field
import logging
import sys
from frozendict import frozendict
import collections
import itertools as it
import re
import json
import csv
from io import TextIOBase
from pathlib import Path
from zensols.config import Dictable
from zensols.persist import (
    persisted, PersistableContainer, PersistedWork,
    OneShotFactoryStash, ReadOnlyStash, PrimeableStash,
)
from zensols.cli import ApplicationError
from zensols.nlp import (
    LexicalSpan, TextContainer, FeatureToken, FeatureSentence, FeatureDocument,
    FeatureDocumentParser,
)
from . import AnnotationError

logger = logging.getLogger(__name__)


@dataclass
class AnnotationSection(Dictable):
    """An annotated section that lexically demarcates a section and identifies it.

    """
    id_ann: int = field()
    """The unique (Inception disambiguation) ID."""

    id: str = field()
    """The section body identifier."""

    header_spans: Tuple[LexicalSpan] = field()
    """Spans that identify header text."""

    body_span: LexicalSpan = field()
    """Identifies the extent of the identified section body."""


@dataclass
class AnnotationNote(PersistableContainer, Dictable):
    """An annotated note with identification information and annotated sections.

    """
    _PERSITABLE_METHODS = {'_get_annotation'}
    _DICTABLE_ATTRIBUTES = {'sections', 'category'}

    hadm_id: int = field()
    """The MimicIII hospital admission ID."""

    row_id: int = field()
    """The MimicIII note (event) ID."""

    anon_path: Path = field()
    """The path to the JSON annotation file."""

    note_path: Path = field()
    """The path to the extracted source MimicIII note event text."""

    def __post_init__(self):
        super().__init__()
        self._sections = PersistedWork('_sections', self, transient=True)
        self._category = None
        self._age_type = None

    @classmethod
    def from_prediction(cls, sections: Tuple[AnnotationSection]):
        inst = cls(None, None, None, None)
        inst._category = FeatureToken.NONE
        inst._sections.set(sections)
        return inst

    @persisted('_annotation')
    def _get_annotation(self) -> Dict[str, Any]:
        with open(self.anon_path) as f:
            anon = json.load(f)
        assert self.hadm_id == anon['hadm_id']
        assert self.row_id == anon['row_id']
        return anon

    def _create_span(self, span: Dict[str, int]) -> LexicalSpan:
        return LexicalSpan(span['begin'], span['end'])

    def _create_sec(self, sec: Dict[str, Any]):
        return AnnotationSection(
            sec['id_ann'], sec['id'],
            header_spans=tuple(map(self._create_span, sec['header_spans'])),
            body_span=self._create_span(sec['body_span']))

    @property
    @persisted('_sections')
    def sections(self) -> Tuple[AnnotationSection]:
        """The annotated sections of the note."""
        return tuple(map(self._create_sec,
                         self._get_annotation()['sections']))

    @property
    @persisted('_sections_by_id', transient=True)
    def sections_by_id(self) -> Dict[str, AnnotationSection]:
        return frozendict({x[0]: x[1] for x in self.sections.items()})

    @property
    def content(self) -> str:
        """The source MimicIII note event text."""
        with open(self.note_path) as f:
            return f.read()

    @property
    def category(self) -> str:
        """The category of the note (i.e. ``Discharge summary``)."""
        if self._category is None:
            self._category = self._get_annotation()['category']
        return self._category

    @property
    def age_type(self) -> str:
        if not hasattr(self, '_age_type') or self._age_type is None:
            self._age_type = self._get_annotation()['age_type']
        return self._age_type

    def __str__(self) -> str:
        ns = len(self.sections)
        return f'{self.hadm_id}-{self.row_id}-{self.category}: {ns} sections'

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class AnnotationStash(OneShotFactoryStash):
    """A stash that creates :clas:`.AnnotationNote` instances from the JSON
    annotations and extracted MimicIII note event text.

    """
    _FILE_ID_REGEX = re.compile(r'^(\d+)-(\d+)-(.+)$')

    anon_path: Path
    notes_text_dir: Path

    def _assert_files(self):
        if not self.anon_path.is_dir():
            raise ApplicationError(
                f'Missing annotations directory: {self.anon_path}')
        if not self.notes_text_dir.parent.exists:
            raise ApplicationError(
                f'Missing parent notes parent: {self.notes_text_dir}')
        self.notes_text_dir.mkdir(exist_ok=True)

    def _annotation_note(self, path: Path) -> AnnotationNote:
        m: re.Match = self._FILE_ID_REGEX.match(path.stem)
        if m is None:
            raise AnnotationError(
                f'Invalid annotation file name: {path}')
        hadm_id, row_id, category = m.groups()
        hadm_id, row_id = int(hadm_id), int(row_id)
        note_file = self.notes_text_dir / f'{row_id}.txt'
        return AnnotationNote(hadm_id, row_id, path, note_file)

    def _get_worker_type(self):
        return 'm'

    def worker(self) -> Iterable[Tuple[str, AnnotationNote]]:
        self._assert_files()
        logger.debug(f'reading row IDs from: {self.anon_path}')
        for apath in self.anon_path.iterdir():
            afile: AnnotationNote = self._annotation_note(apath)
            yield (str(afile.row_id), afile)


@dataclass
class MimicNotesExtractor(object):
    """This extracts notes from the ``NOTEEVENTS.csv`` to a directory with each
    note in a file named by its respective ``row_id``.

    """
    anon_stash: AnnotationStash = field()
    """Creates :class:`.AnnotationNote` instances having the annotation and
    location information.

    """
    notes_file: Path = field()
    """The path to the MimicIII ``NOTEEVENTS.csv`` file."""

    limit: int = field(default=sys.maxsize)
    """The max number of notes to extract."""

    def _assert_files(self):
        if not self.notes_file.is_file():
            raise ApplicationError(
                f'Missing MIMIC III notes: {self.notes_file}')

    def _write_note(self, path: Path, text: str):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'writing {path}')
        with open(path, 'w') as f:
            f.write(text)

    @property
    def is_extracted(self) -> bool:
        if len(self.anon_stash) > 0:
            an: AnnotationNote = next(iter(self.anon_stash.values()))
            return an.note_path.is_file()
        return False

    def __call__(self):
        self._assert_files()
        n_files = len(self.anon_stash)
        logger.info(f'extracting {n_files} annotation files')
        extracted = 0
        with open(self.notes_file) as csvfile:
            cr = csv.reader(csvfile)
            header = next(cr)
            header = dict(zip(header, it.count()))
            row_id_col, text_col = header['ROW_ID'], header['TEXT']
            for row in it.islice(cr, self.limit):
                row_id = row[row_id_col]
                afile: AnnotationNote = self.anon_stash.get(row_id)
                if afile is not None:
                    text = row[text_col]
                    self._write_note(afile.note_path, text)
                    extracted += 1
                if extracted >= n_files:
                    break
        logger.info(f'extracted {extracted} mimic note events')


@dataclass
class NoteFeatureSentence(FeatureSentence):
    @property
    def annotations(self):
        return tuple(map(lambda t: t.section_id_, self.token_iter()))


@dataclass
class NoteFeatureDocument(FeatureDocument):
    _PERSITABLE_PROPERTIES = {'anon_note'}

    anon_note: AnnotationNote = field(default=None)

    def __post_init__(self):
        super().__post_init__()
        #assert self.anon_note is not None

    @property
    def category(self) -> str:
        """The category of the note, such as ``Discharge summary``."""
        if self.anon_note is None:
            return 'Discharge summary'
        else:
            return self.anon_note.category

    def get_section_tokens(self, sec: AnnotationSection) -> \
            Tuple[FeatureToken]:
        """Get tokens for the given section.

        :param sec: the query section

        :raises KeyError: if the section is not in the document
        """
        ixs: Dict[str, List[int]] = self._sec2idx.get(sec.id_ann)
        tokens_by_idx: Dict[int, FeatureToken] = self.tokens_by_idx
        if ixs is not None:
            return tuple(map(lambda i: tokens_by_idx[i], ixs))

    def _combine_documents(self, docs: Tuple[FeatureDocument],
                           cls: Type[FeatureDocument],
                           concat_tokens: bool = True) -> FeatureDocument:
        return super()._combine_documents(
            docs, cls, concat_tokens, anon_note=self.anon_note)

    def _map_anons(self):
        """Add section ID and header features to tokens and map sections to tokens for
        analysis.

        """
        def containing_section(span: LexicalSpan):
            sec: AnnotationSection
            for sec in na.sections:
                bspan: LexicalSpan = sec.body_span
                if span.overlaps_with(bspan, inclusive=False):
                    return sec

        sec2idx: Dict[int, List[int]] = collections.defaultdict(list)
        na: AnnotationNote = self.anon_note
        tok: FeatureToken
        for tok in self.token_iter():
            tok_span: LexicalSpan = tok.lexspan
            is_header: bool = False
            sec: AnnotationSection = containing_section(tok_span)
            if sec is not None:
                section_id = sec.id
                is_header = any(map(
                    lambda s: s.overlaps_with(tok_span, inclusive=False),
                    sec.header_spans))
                sec2idx[sec.id_ann].append(tok.idx)
            else:
                section_id: str = FeatureToken.NONE
            tok.section_id_ = section_id
            tok.is_header = is_header
        self._sec2idx = sec2idx

    def write_headers(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        anote: AnnotationNote = self.anon_note
        sec: AnnotationSection
        for sec in anote.sections:
            self._write_line(f'{("-"*40)}{sec.id}{("-"*40)}', depth, writer)
            body: str = self.text[sec.body_span.begin:sec.body_span.end]
            self._write_block(body, depth, writer)

    def __repr__(self):
        return TextContainer.__repr__(self)


@dataclass
class NoteFeatureDocumentStash(ReadOnlyStash, PrimeableStash):
    mimic_notes_extractor: MimicNotesExtractor
    doc_parser: FeatureDocumentParser
    anon_stash: AnnotationStash

    def load(self, name: str) -> NoteFeatureDocument:
        anon_note: AnnotationNote = self.anon_stash[name]
        doc = self.doc_parser(anon_note.content, anon_note=anon_note)
        doc._map_anons()
        return doc

    def keys(self) -> Iterable[str]:
        return self.anon_stash.keys()

    def exists(self, name: str) -> bool:
        return self.anon_stash.exists(name)

    def prime(self):
        super().prime()
        extracted = self.mimic_notes_extractor.is_extracted
        if not extracted:
            self.mimic_notes_extractor()
