"""Contains sectioned document batch classes and mappings.

"""
__author__ = 'Paul Landes'

from typing import Tuple, Type, Any, List
from dataclasses import dataclass, field
import logging
import copy as cp
from zensols.config import Settings
from zensols.persist import persisted
from zensols.deeplearn.batch import (
    DataPoint,
    Batch,
    BatchStash,
    ManagerFeatureMapping,
    FieldFeatureMapping,
    BatchFeatureMapping,
)
from zensols.nlp import FeatureSentence, FeatureDocument
from zensols.deeplearn.result import ResultsContainer
from zensols.deepnlp.classify import ClassificationPredictionMapper
from . import NoteFeatureDocument

logger = logging.getLogger(__name__)


@dataclass
class SectionPredictionMapper(ClassificationPredictionMapper):
    def _create_data_point(self, cls: Type[DataPoint],
                           feature: Any) -> DataPoint:
        return cls(None, self.batch_stash, feature, True)

    def _create_features(self, sent_text: str) -> Tuple[FeatureSentence]:
        doc: FeatureDocument = self.vec_manager.parse(
            sent_text, anon_note=None)
        self._docs.append(doc)
        return [doc]

    def map_results(self, result: ResultsContainer) -> Settings:
        classes = self._map_classes(result)
        return Settings(classes=tuple(classes), docs=tuple(self._docs))


@dataclass
class SectionDataPoint(DataPoint):
    doc: NoteFeatureDocument = field(repr=False)
    is_pred: bool = field(default=False)

    @property
    @persisted('_section_ids', transient=True)
    def section_ids(self) -> Tuple[str]:
        if self.is_pred:
            return tuple([None] * self.doc.token_len)
        else:
            r = tuple(map(lambda t: t.section_id_, self.doc.token_iter()))
            return r

    @property
    def note_category(self) -> List[str]:
        if self.doc is None:
            return ['Discharge summary']
        else:
            return [self.doc.category]

    @property
    def trans_doc(self) -> FeatureDocument:
        """The document used by the transformer vectorizers.  Return ``None`` for
        prediction data points to avoid vectorization.

        """
        if self.is_pred:
            return None
        return self.doc

    def __len__(self):
        return self.doc.token_len


@dataclass
class SectionBatch(Batch):
    LANGUAGE_FEATURE_MANAGER_NAME = 'language_feature_manager'
    GLOVE_50_EMBEDDING = 'glove_50_embedding'
    GLOVE_300_EMBEDDING = 'glove_300_embedding'
    GLOVE_300_EMBEDDING = 'glove_300_embedding'
    WORD2VEC_300_EMBEDDING = 'word2vec_300_embedding'
    FASTTEXT_CRAWL_300_EMBEDDING = 'fasttext_crawl_300_embedding'
    TRANSFORMER_EMBEDDING = 'transformer_fixed_embedding'
    TRANSFORMER_LABEL = 'section_ids_trans'
    MAJOR_SENT_LABEL = 'majorsent_ids'
    TRANSFORMER_MAJORSENT_EMBEDDING = 'transformer_majorsent_trainable_embedding'
    TRANSFORMER_MAJORSENT_FIXED_EMBEDDING = 'transformer_majorsent_fixed_embedding'
    TRANSFORMER_MAJORSENT_FIXED_BIOBERT_EMBEDDING = 'transformer_majorsent_fixed_biobert_embedding'
    EMBEDDING_ATTRIBUTES = {
        GLOVE_50_EMBEDDING,
        GLOVE_300_EMBEDDING,
        WORD2VEC_300_EMBEDDING,
        FASTTEXT_CRAWL_300_EMBEDDING,
        #TRANSFORMER_EMBEDDING,
        TRANSFORMER_MAJORSENT_EMBEDDING,
        TRANSFORMER_MAJORSENT_FIXED_EMBEDDING,
        TRANSFORMER_MAJORSENT_FIXED_BIOBERT_EMBEDDING,
    }
    MAPPINGS = BatchFeatureMapping(
        'section_ids',
        [ManagerFeatureMapping(
            'token_classify_label_vectorizer_manager',
            (FieldFeatureMapping('section_ids', 'tclabel', True, is_label=True),
             FieldFeatureMapping('mask', 'tcmask', True, 'section_ids'),

             FieldFeatureMapping(TRANSFORMER_LABEL, 'tclabeltrans', True, 'trans_doc', is_label=True),
             FieldFeatureMapping('masktrans', 'tcmasktrans', True, 'trans_doc', is_label=True),

             FieldFeatureMapping(MAJOR_SENT_LABEL, 'mslabel', True, 'trans_doc', is_label=True),
             FieldFeatureMapping('majorsent_id_mask', 'msmask', True, 'trans_doc', is_label=True),
             )),
         ManagerFeatureMapping(
             'note_vectorizer_manager',
             (FieldFeatureMapping('note_category', 'notecat'),)),
         ManagerFeatureMapping(
             LANGUAGE_FEATURE_MANAGER_NAME,
             (FieldFeatureMapping('note_category_token', 'notecattok', True, 'doc'),
              FieldFeatureMapping(GLOVE_50_EMBEDDING, 'wvglove50', True, 'doc'),
              FieldFeatureMapping(GLOVE_300_EMBEDDING, 'wvglove300', True, 'doc'),
              FieldFeatureMapping(WORD2VEC_300_EMBEDDING, 'w2v300', True, 'doc'),
              FieldFeatureMapping(FASTTEXT_CRAWL_300_EMBEDDING, 'wvftcrawl300', True, 'doc'),
              FieldFeatureMapping(TRANSFORMER_EMBEDDING, 'transformer_sent_fixed', True, 'doc'),
              FieldFeatureMapping(TRANSFORMER_MAJORSENT_EMBEDDING, 'mstrantrainable', True, 'doc'),
              FieldFeatureMapping(TRANSFORMER_MAJORSENT_FIXED_EMBEDDING, 'mstranfixed', True, 'doc'),
              FieldFeatureMapping(TRANSFORMER_MAJORSENT_FIXED_BIOBERT_EMBEDDING, 'mstranfixed', True, 'doc'),
              ),)])

    TRANS_MAPPINGS = cp.deepcopy(MAPPINGS)
    TRANS_MAPPINGS.label_attribute_name = TRANSFORMER_LABEL

    MAJOR_SENT_FIXED_MAPPINGS = cp.deepcopy(MAPPINGS)
    MAJOR_SENT_FIXED_MAPPINGS.label_attribute_name = MAJOR_SENT_LABEL

    def _get_batch_feature_mappings(self) -> BatchFeatureMapping:
        stash: BatchStash = self.batch_stash
        if self.TRANSFORMER_LABEL in stash.decoded_attributes:
            maps = self.TRANS_MAPPINGS
        elif self.MAJOR_SENT_LABEL in stash.decoded_attributes:
            maps = self.MAJOR_SENT_FIXED_MAPPINGS
        else:
            maps = self.MAPPINGS
        return maps
