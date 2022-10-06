"""Majority label across sentences.

"""
__author__ = 'Paul Landes'

from typing import Sequence, Any, Tuple, List, Union
from dataclasses import dataclass, field
import logging
import collections
from sklearn.preprocessing import LabelEncoder
import torch
from torch import Tensor
from zensols.persist import persisted
from zensols.nlp import FeatureDocument, FeatureSentence
from zensols.deeplearn import TorchConfig
from zensols.deeplearn.vectorize import (
    FeatureContext,
    TensorFeatureContext,
    NonUniformDimensionEncoder,
    NominalEncodedEncodableFeatureVectorizer,
)
from zensols.deeplearn.batch import Batch
from zensols.deeplearn.model import SequenceNetworkContext
from zensols.deepnlp.vectorize import (
    TextFeatureType, FoldingDocumentVectorizer,
)
from zensols.deepnlp.transformer import TransformerEmbeddingFeatureVectorizer
from zensols.deepnlp.layer import (
    EmbeddedRecurrentCRFSettings,
    EmbeddedRecurrentCRF,
)

logger = logging.getLogger(__name__)


@dataclass
class MajorSentLabelMapper(object):
    section_ids_attribute: str = field()
    majorsent_ids_attribute: str = field()

    def __call__(self, batch: Batch, pred_labels: Tensor = None) -> \
            List[List[int]]:
        tok_labs = []
        for bix, dp in enumerate(batch.data_points):
            fdoc: FeatureDocument = dp.doc
            fdoc_len: int = len(fdoc)
            fdoc_tlen: int = fdoc.token_len
            batch_pred_labels = pred_labels[bix][:fdoc_len]
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'majority label ids: {batch_pred_labels.shape}, ' +
                             f'sentences: {fdoc_len}, ' +
                             f'token lenth {fdoc_tlen}')
            for six, sent in enumerate(fdoc.sents):
                slen = len(sent)
                msid: int = batch_pred_labels[six].item()
                tok_labs.append([msid] * slen)
        return tok_labs


@dataclass
class MajorSentLabelVectorizer(FoldingDocumentVectorizer):
    """Majority label across sentences vectroizer.
    """
    DESCRIPTION = 'majority label'
    FEATURE_TYPE = TextFeatureType.NONE

    delegate_feature_id: str = field(default=None)
    """The feature ID for the aggregate encodeable feature vectorizer."""

    annotations_attribute: str = field(default='annotations')
    """The attribute used to get the features from the
    :class:`~zensols.nlp.FeatureSentence`.  For example,
    :class:`~zensols.nlp.TokenAnnotatedFeatureSentence` has an ``annotations``
    attribute.

    """
    def _get_shape(self) -> Tuple[int]:
        return -1, -1

    def _get_attributes(self, sent: FeatureSentence) -> Sequence[Any]:
        return getattr(sent, self.annotations_attribute)

    @property
    @persisted('_delegate', allocation_track=False)
    def delegate(self) -> NominalEncodedEncodableFeatureVectorizer:
        """The delegates used for encoding and decoding the lingustic features.

        """
        return self.manager[self.delegate_feature_id]

    def _create_decoded_pad(self, shape: Tuple[int]) -> Tensor:
        return self.torch_config.cross_entropy_pad(shape).type(torch.long)

    def _encode(self, doc: FeatureDocument) -> FeatureContext:
        delegate: NominalEncodedEncodableFeatureVectorizer = self.delegate
        le: LabelEncoder = delegate.label_encoder
        noms: List[int] = []
        ids = getattr(doc.sents[0], self.annotations_attribute)
        cnts = collections.defaultdict(lambda: 0)
        for id in ids:
            cnts[id] += 1
        top = sorted(cnts.items(), key=lambda x: x[1], reverse=True)
        lab: str = top[0][0]
        nom: int = le.transform([lab])[0]
        noms.append(nom)
        arr: Tensor = self.torch_config.singleton(noms)
        arr = arr.unsqueeze(0)
        arr = arr.unsqueeze(1)
        return TensorFeatureContext(self.feature_id, arr)


@dataclass
class MajorSentEmbeddingVectorizer(TransformerEmbeddingFeatureVectorizer):
    def _combine_documents(self, docs: Tuple[FeatureDocument]) -> \
            FeatureDocument:
        return docs

    def _encode(self, docs: Tuple[FeatureDocument]) -> FeatureContext:
        de = NonUniformDimensionEncoder(self.torch_config)
        arrs = []
        for doc in docs:
            ctx = super()._encode(doc)
            arrs.append(ctx.document.tensor)
        arr: Tensor = de.encode(arrs)
        return TensorFeatureContext(self.feature_id, arr)

    def _decode(self, context: TensorFeatureContext) -> Tensor:
        return context.tensor


@dataclass
class MajorSentTransformerEmbeddingFeatureVectorizer(TransformerEmbeddingFeatureVectorizer):
    def _decode_sentence(self, sent_ctx: FeatureContext) -> Tensor:
        arr: Tensor = super()._decode_sentence(sent_ctx)
        arr = arr.unsqueeze(1)
        return arr


@dataclass
class MajorSentNetworkSettings(EmbeddedRecurrentCRFSettings):
    label_mapper: MajorSentLabelMapper = field(default=None)
    non_uniform: bool = field(default=False)

    def get_module_class_name(self) -> str:
        return __name__ + '.MajorSent'


class MajorSent(EmbeddedRecurrentCRF):
    MODULE_NAME = 'major sent fixed'

    def _map_labels(self, batch: Batch, context: SequenceNetworkContext,
                    labels: Union[List[List[int]], Tensor]) -> List[List[int]]:
        tok_labs = []
        is_tensor = isinstance(labels, Tensor)
        for bix, dp in enumerate(batch.data_points):
            fdoc: FeatureDocument = dp.doc
            fdoc_len: int = len(fdoc)
            fdoc_tlen: int = fdoc.token_len
            if is_tensor:
                batch_labels = labels[bix][:fdoc_len]
            else:
                batch_labels = labels[bix]
            if logger.isEnabledFor(logging.DEBUG):
                self._shape_or_list_debug('majority label ids', batch_labels)
                logger.debug(f'sentences: {fdoc_len}, token lenth {fdoc_tlen}')
            for six, sent in enumerate(fdoc.sents):
                slen = len(sent)
                msid: int = batch_labels[six]
                if is_tensor:
                    tok_labs.append(msid.repeat((slen,)))
                else:
                    tok_labs.append([msid] * slen)
        if is_tensor:
            tok_labs = torch.cat(tok_labs, dim=0)
        return tok_labs

    def forward_embedding_features(self, batch: Batch) -> Tensor:
        """Use the embedding layer return the word embedding tensors.

        """
        if self.net_settings.non_uniform:
            tc: TorchConfig = batch.torch_config
            self._debug('forward embedding')
            x = batch.attributes[self.embedding_attribute_name]
            self._shape_debug('non-uniform', x)
            de = NonUniformDimensionEncoder(tc)
            docs: Tuple[Tensor] = de.decode(x)
            arrs: List[Tensor] = []
            if self.logger.isEnabledFor(logging.DEBUG):
                self._debug(f'decoded {len(docs)} docs')
            for doc in docs:
                self._shape_debug('forward doc', doc)
                x = self.embedding(doc)
                self._shape_debug('decoded embedding', x)
                arrs.append(x)
            batch_max = max(map(lambda t: t.size(0), arrs))
            x_embed = tc.zeros((len(batch), batch_max, self.embedding_dimension))
            self._shape_debug('padded embedded', x_embed)
            for bix, arr in enumerate(arrs):
                x_embed[bix, :arr.size(0)] = arr
            return x_embed
        else:
            return super().forward_embedding_features(batch)
