"""Facade client entry for the model.

"""
__author__ = 'Paul Landes'

from typing import Tuple, Iterable, List, Any
from dataclasses import dataclass
import logging
from zensols.config import Settings
from zensols.nlp import FeatureToken, LexicalSpan
from zensols.deepnlp.classify import TokenClassifyModelFacade
from . import AnnotationSection, AnnotationNote, NoteFeatureDocument

logger = logging.getLogger(__name__)


@dataclass
class SectionFacade(TokenClassifyModelFacade):
    def _configure_debug_logging(self):
        from zensols.cli import LogConfigurator

        LogConfigurator.set_format('%(asctime)s:%(message)s')
        super()._configure_debug_logging()
        for i in [__name__, 'sidmodel.majorsent']:
            logging.getLogger(i).setLevel(logging.DEBUG)
        for i in ['zensols.deepnlp.transformer.vectorizers']:
            logging.getLogger(i).setLevel(logging.WARNING)

    def get_predictions_factory(self, *args, **kwargs):
        fac = super().get_predictions_factory(*args, **kwargs)
        if 'majorsent_ids' in self.batch_stash.decoded_attributes:
            fac.label_vectorizer_name = 'token_classify_label_1_vectorizer'
        return fac

    def predict(self, sents: Iterable[str]) -> Any:
        pred: Settings = super().predict(sents)
        docs: Tuple[NoteFeatureDocument] = pred.docs
        classes: Tuple[str] = pred.classes
        doc_tok_lists: List[Tuple[str, List[FeatureToken]]] = []
        for labels, doc in zip(classes, docs):
            last_lab: str = None
            tok_list: Tuple[str, List[FeatureToken]] = None
            tok_lists: List[Tuple[str, List[FeatureToken]]] = []
            doc_tok_lists.append(tok_lists)
            label: str
            tok: FeatureToken
            for label, tok in zip(labels, doc.token_iter()):
                if last_lab != label:
                    if tok_list is not None:
                        tok_lists.append((label, tok_list))
                    tok_list = [tok]
                else:
                    tok_list.append(tok)
                last_lab = label
            if tok_list is not None and len(tok_list) > 0:
                tok_lists.append((label, tok_list))
        ann_id = 0
        for doc, tok_lists in zip(docs, doc_tok_lists):
            secs: List[AnnotationSection] = []
            tok_lists = tuple(
                filter(lambda x: x[0] != FeatureToken.NONE, tok_lists))
            for label, toks in tok_lists:
                span: LexicalSpan = None
                if len(toks) == 1:
                    span = toks[0].lexspan
                else:
                    begin = toks[0].lexspan.begin
                    end = toks[-1].lexspan.end
                    span = LexicalSpan(begin, end)
                assert span is not None
                secs.append(AnnotationSection(ann_id, label, (), span))
                ann_id += 1
            doc.anon_note = AnnotationNote.from_prediction(secs)
            doc.tok_lists = tok_lists
        return docs
