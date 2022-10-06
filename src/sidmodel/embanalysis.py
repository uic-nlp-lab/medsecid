from __future__ import annotations
"""Data analysis on section identification.

"""
__author__ = 'Paul Landes'

from typing import Dict, List, Tuple, Set, Sequence, Union
from dataclasses import dataclass, field
import sys
import logging
from itertools import chain
import itertools as it
import collections
import math
from frozendict import frozendict
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
import plotly.express as px
from zensols.persist import Stash, persisted, PersistedWork
from zensols.nlp import FeatureToken, FeatureDocument, FeatureSentence
from zensols.mednlp import MedicalLibrary
from zensols.dataset import (
    OutlierDetector, DimensionReducer, DecomposeDimensionReducer
)
from zensols.deepnlp.embed import TextWordEmbedModel
from . import (
    AnnotationNote, AnnotationSection, NoteFeatureDocument,
)

logger = logging.getLogger(__name__)


@dataclass
class CuiDataPoint(object):
    token: FeatureToken
    vec: np.ndarray = field(repr=False)
    key: str
    count: int = field(default=1)
    section: str = field(default=None)
    category: str = field(default=None)
    age_type: str = field(default=None)

    def __post_init__(self):
        self._hash = hash(self.key)

    @staticmethod
    def create_key(token: FeatureToken) -> str:
        return token.norm + '.' + token.cui_

    @property
    def cui_(self) -> str:
        return self.token.cui_

    def copy(self) -> CuiDataPoint:
        p = self.__class__(self.token, self.vec, self.key, self.count,
                           self.section, self.category, self.age_type)
        if hasattr(self, 'tfidf'):
            p.tfidf = self.tfidf
        p._hash = self._hash
        return p

    def inc(self, point: CuiDataPoint = None):
        if point is None:
            self.count += 1
        else:
            self.count += point.count

    def __eq__(self, other):
        return self.key == other.key

    def __hash__(self):
        return self._hash

    def __str__(self):
        t = self.token
        s = t.norm
        if t.is_concept and t.pref_name_ != t.norm:
            cnt = ''
            if self.count > 1:
                cnt = f': {self.count}'
            s = f'{s}: {t.pref_name_}' + cnt
        return s

    def __repr__(self):
        return self.__str__()


@dataclass
class EmbeddingAnalyzer(object):
    doc_stash: Stash
    med_lib: MedicalLibrary

    normalize: str = field()
    """One of ``unit``, ``standardize`` or ``None``."""

    dim_reduction_meth: str = field()
    """Method to dimensionaly reduce the data: pca, svd, tnse"""

    width: int = field()
    """The width of the ployly render element."""

    height: int = field()
    """The height of the ployly render element."""

    axis_range: int = field(default=None)
    """The X, Y, [Z] ranges of the ployly render element.  Seems to ignore Z.

    """
    add_mean: bool = field(default=True)
    """Add a mean/average ball to the data."""

    doc_limit: int = field(default=sys.maxsize)
    """One of pca, svd or tsne."""

    outlier_proportion: float = field(default=None)
    """The proportion of outliers to remove.  The higher this is, the more data
    points are removed.

    """
    calc_per_section: bool = field(default=False)
    """Whether to reduce dimension and remove outliers on a per section basis.

    """
    max_point_size: int = field(default=30)
    """The largest Ployly data point size for visualization."""

    mean_point_size: int = field(default=23)
    """The average Ployly data point size for visualization."""

    start_size: int = field(default=8)
    """The starting Ployly data point size for visualization."""

    plot_type: str = field(default='section')
    """The kind of plot, which is one of: ``section``, ``age``, ``shared`` or
    ``rand``.

    """
    plot_tfidf: bool = field(default=True)
    """Whether or not to plot TF/IDF or counts as a size."""

    tfidf_threshold: int = field(default=0.05)
    """The threshold used to elminite data points with lower TF/IDF values when
    plotting TF/IDF (see :obj:`plot_tfidf`).

    """
    plot_components: int = field(default=sys.maxsize)
    """The number PCA components to plot when using PCA."""

    keep_age_types: Set[str] = frozenset({'adult', 'newborn', 'pediatric'})
    """The types of data to keep across age."""

    tfidf_sections: Set[str] = None
    """Sections to keep when using a TF/IDF plot (see :obj:`plot_tfidf`)."""

    ontology_file: Path = field(default=None)
    """Used for the radiology vs discharge summary analysis."""

    temporary_dir: Path = field(default=None)
    """Where to put cached files and the temporary stats."""

    def __post_init__(self):
        self._doc_points_pw = PersistedWork(
            self.temporary_dir / 'doc-points.dat', self,
            cache_global=True, mkdir=True)
        self.expected_vars = self.temporary_dir / 'expected-vars.csv'
        self._agg_doc_points_pw = PersistedWork(
            '_agg_doc_points_pw', self, cache_global=True)
        self._agg_max_point_count_pw = PersistedWork(
            '_agg_max_point_count_pw', self, cache_global=True)

    @persisted('_doc_points_pw')
    def _doc_points(self) -> Dict[str, Dict[str, CuiDataPoint]]:
        """
        :return: a dictionary of note, section ID, CUI data points
        """
        doc_points = {}
        emb: TextWordEmbedModel = self.med_lib.cui2vec_embedding
        doc: NoteFeatureDocument
        for row_id, doc in it.islice(self.doc_stash, self.doc_limit):
            sec2points: Dict[str, Dict[str, CuiDataPoint]] = \
                collections.defaultdict(dict)
            doc_points[row_id] = sec2points
            anon_note: AnnotationNote = doc.anon_note
            cat = anon_note.category
            sec: AnnotationSection
            for sec in anon_note.sections:
                toks = doc.get_section_tokens(sec)
                if toks is None:
                    continue
                for tok in filter(lambda t: t.is_concept and len(t.norm.strip()) > 2, toks):
                    vec: np.ndarray = emb.get(tok.cui_)
                    if vec is not None:
                        key = CuiDataPoint.create_key(tok)
                        cui_point = sec2points[sec.id].get(key)
                        if cui_point is None:
                            point = CuiDataPoint(
                                tok, vec, key, category=cat,
                                section=sec.id, age_type=anon_note.age_type)
                            sec2points[sec.id][key] = point
                        else:
                            cui_point.inc()
        return frozendict(doc_points)

    @persisted('_all_points_pw')
    def _all_points(self) -> Tuple[CuiDataPoint]:
        sec2points: Dict[str, Dict[str, CuiDataPoint]] = self._agg_doc_points()
        return tuple(chain.from_iterable(
            map(lambda d: d.values(), sec2points.values())))

    @persisted('_points_by_age_pw', cache_global=True)
    def _points_by_age(self) -> Dict[str, Dict[str, CuiDataPoint]]:
        by_age = collections.defaultdict(dict)
        points: Tuple[CuiDataPoint] = self._all_points()
        for point in points:
            by_cui = by_age[point.age_type]
            cp_point: CuiDataPoint = by_cui.get(point.key)
            if cp_point is None:
                by_cui[point.key] = point.copy()
            else:
                cp_point.inc()
        return frozendict({x[0]: frozendict(x[1]) for x in by_age.items()})

    @persisted('_agg_doc_points_pw')
    def _agg_doc_points(self, limit: int = sys.maxsize) -> \
            Dict[str, List[CuiDataPoint]]:
        """
        :return: ection ID, CUI data points
        """
        s2p: Dict[str, Dict[str, CuiDataPoint]] = collections.defaultdict(dict)
        for dps in it.islice(self._doc_points().values(), limit):
            k: str
            vd: Dict[str, Dict[str, CuiDataPoint]]
            for k, vd in dps.items():
                for point in vd.values():
                    p = s2p[k].get(point.key)
                    if p is None:
                        s2p[k][point.key] = point.copy()
                    else:
                        p.inc(point)
        return frozendict({x[0]: frozendict(x[1]) for x in s2p.items()})

    @persisted('_agg_max_point_count_pw')
    def _agg_max_point_count(self) -> int:
        return max(chain.from_iterable(
            map(lambda pd: map(lambda p: p.count, pd.values()),
                self._agg_doc_points().values())))

    @staticmethod
    def feat_to_tokens(docs: Tuple[FeatureDocument]) -> Tuple[str]:
        """Create a tuple of string tokens from a set of documents suitable for
        document indexing.  The strings are the lemmas of the tokens.

        **Important**: this method must remain static since the LSI instance of
        this class uses it as a factory function in the a vectorizer.

        """
        toks = map(lambda d: d.norm.lower(),
                   filter(lambda d: not d.is_stop and not d.is_punctuation,
                          chain.from_iterable(
                              map(lambda d: d.tokens, docs))))
        return tuple(toks)

    @persisted('_tfidf_pw', cache_global=True)
    def _tfidf(self) -> Tuple[Dict[str, List[CuiDataPoint]], float]:
        sec2points = self._agg_doc_points()
        sec: str
        points: Dict[str, CuiDataPoint]
        docs: List[Tuple[str]] = []
        point_sets = []
        for sec, points in sec2points.items():
            words = list(map(lambda p: p.token, points.values()))
            sent = FeatureSentence(words)
            doc = sent.to_document()
            doc.section = sec
            docs.append(doc)
            point_sets.append((sec, points))
        tfidf = TfidfVectorizer(
            tokenizer=self.feat_to_tokens,
            smooth_idf=True,
            lowercase=False)
        X = tfidf.fit_transform(docs)
        tok2id = dict(zip(tfidf.get_feature_names_out(), it.count()))
        for dix, (sec, points) in enumerate(sec2points.items()):
            for p in points.values():
                tix = tok2id.get(p.token.norm)
                if tix is not None:
                    p.tfidf = X[dix, tix]
                else:
                    p.tfidf = 0
        max_tfidf = max(map(lambda p: p.tfidf, chain.from_iterable(
            map(lambda d: d.values(), sec2points.values()))))
        return sec2points, max_tfidf

    def _tfidf_trimmed(self) -> Dict[str, List[CuiDataPoint]]:
        sec2points = self._tfidf()[0]
        trimmed = {}
        for sec, points in sec2points.items():
            if self.tfidf_sections is not None and sec not \
               in self.tfidf_sections:
                continue
            trimmed_points = {}
            trimmed[sec] = trimmed_points
            for k, point in points.items():
                if point.tfidf > self.tfidf_threshold:
                    trimmed_points[k] = point
        return trimmed

    @property
    @persisted('_distance_dataframe', cache_global=True)
    def distance_dataframe(self) -> pd.DataFrame:
        points: Tuple[CuiDataPoint] = self._all_points()
        vecs: Tuple[np.ndarray] = tuple(map(lambda p: p.vec, points))
        vecs: np.ndarray = np.stack(vecs, axis=0)
        mean: np.ndarray = vecs.mean(axis=0)
        dists = np.linalg.norm(vecs - mean, ord=2, axis=1)
        pdists = sorted(zip(points, dists.tolist()), key=lambda x: x[1])
        rows = []
        for point, dist in pdists:
            rows.append((point.section, point.token.norm, point.token.pref_name_, point.age_type, dist))
        df = pd.DataFrame(rows, columns='section token desc age distance'.split())
        return df

    def _partition_radiology_ds(self) -> Tuple[Tuple[str, Set[str]]]:
        sec2points = self._agg_doc_points()
        df = pd.read_csv(self.ontology_file)
        rad = set(df[df['note_name'] == 'Radiology']['section_id'].tolist())
        ds = set(df[df['note_name'] == 'Discharge Summary']['section_id'].tolist())
        keys = set(sec2points.keys())
        rad = frozenset(rad & keys)
        ds = frozenset(ds & keys)
        partition = (('radiology', rad),
                     ('discharge summary', ds))
        return partition

    def _get_points_by_category(self, notes: Tuple[Tuple[str, Set[str]]]) -> \
            Dict[str, Dict[str, CuiDataPoint]]:
        by_note = collections.defaultdict(dict)
        points_by_sec: Dict[str, List[CuiDataPoint]] = self._tfidf_trimmed()
        for note, sids in notes:
            note_points = by_note[note]
            for point_sec, points in points_by_sec.items():
                if point_sec in sids:
                    note_points.update(points)
        shared: Dict[str, CuiDataPoint] = {}
        cnts: Dict[str, int] = collections.defaultdict(lambda: 0)
        note: str
        points: Dict[str, CuiDataPoint]
        for k in chain.from_iterable(map(lambda d: d.keys(), by_note.values())):
            cnts[k] += 1
        for pk, v in cnts.items():
            if v > 1:
                for k, points in by_note.items():
                    point = points.get(pk)
                    if point is not None:
                        shared[pk] = point
        for points in by_note.values():
            for k in shared.keys():
                points.pop(k, None)
        by_note['shared'] = shared
        return by_note

    def _dim_reduce(self, emb_vecs: np.ndarray, dim: int):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'reducing dimension {emb_vecs.shape} -> {dim}')
        cls = DimensionReducer
        if DecomposeDimensionReducer.is_decompose_method(self.dim_reduction_meth):
            cls = DecomposeDimensionReducer
        dim_reducer = cls(emb_vecs, dim, self.dim_reduction_meth,
                          self.normalize)
        self.dim_reducers = (dim_reducer,)
        return dim_reducer.reduced

    @property
    def model(self) -> Union[PCA, TruncatedSVD, TSNE]:
        return self.dim_reducers[0].model

    def _plotly_init(self):
        import warnings
        m = 'The PCA initialization in TSNE will change to have the standard*'
        warnings.filterwarnings("ignore", message=m)
        # configure Plotly to be rendered inline in the notebook.
        plotly.offline.init_notebook_mode()

    def _rand_3d_data(self, dim: int = 3, samples: int = 100,
                      add_curve: bool = False) -> Tuple[CuiDataPoint]:
        def map_rand(v: np.ndarray):
            p = CuiDataPoint(FeatureToken(0, 0, 0, 'n'), v, 'n', 'n', 'n', 'n')
            p.tfidf = 0.1
            p.token.is_concept = False
            return p

        r = np.arange(samples, dtype=float)
        rdata = np.stack([r]*dim).T
        rdata += np.random.normal(size=rdata.shape) * 1.5
        if add_curve:
            por = int(len(rdata) * .5)
            rdata[por:, 0] += (np.arange(por) ** 1.2)
        return tuple(map(map_rand, rdata))

    def _plotly_embeds(self, dim: int):
        np.random.seed(seed=1)

        def map_size(p: CuiDataPoint) -> int:
            if self.plot_tfidf:
                return int(p.tfidf * 100)
            else:
                return int(math.log(p.count + 1) * div)

        sec2points: Dict[str, Dict[str, CuiDataPoint]]
        if self.plot_tfidf:
            sec2points = self._tfidf_trimmed()
        else:
            sec2points = self._agg_doc_points()
        max_size = self.max_point_size
        div = max_size / math.log(max_size)
        if self.plot_type == 'age':
            by_age = self._points_by_age()
            labs = sorted(list(by_age))
        elif self.plot_type == 'section' or self.plot_type == 'rand':
            labs = sorted(sec2points.keys())
        elif self.plot_type == 'shared':
            secs = self._partition_radiology_ds()
            by_cat = self._get_points_by_category(secs)
            labs = sorted(list(by_cat.keys()))
        else:
            raise ValueError(f'Unknown plot type: {self.plot_type}')
        colors = px.colors.sequential.Turbo
        embs = []
        start = 0
        point_lim = dim - 1
        color_ixs = tuple(range(len(labs)))
        self.dim_reducers = []
        for lab, color_ix in zip(labs, color_ixs):
            points: Sequence[CuiDataPoint]
            if self.plot_type == 'age':
                color_ix = (color_ix * 3) + 1
            elif self.plot_type == 'section' or self.plot_type == 'rand':
                color_ix = ((color_ix * 3) + 1) % len(colors)
            elif self.plot_type == 'shared':
                color_ix = (color_ix * 5) + 1
            color = colors[color_ix]
            if self.plot_type == 'age':
                points = by_age[lab].values()
            elif self.plot_type == 'section' or self.plot_type == 'rand':
                points = sec2points[lab].values()
            elif self.plot_type == 'shared':
                points = by_cat[lab].values()
            if self.plot_type == 'rand':
                points = self._rand_3d_data()
            else:
                age_types = self.keep_age_types
                points = tuple(filter(lambda p: p.age_type in age_types, points))
            vecs = None
            if self.calc_per_section:
                vecs = np.stack(tuple(map(lambda p: p.vec, points)))
                import warnings
                warnings.filterwarnings(
                    'ignore', message='invalid value encountered in sqrt')
                logger.debug(f'initial outlier detection shape: {vecs.shape}')
                dim_reducer = DecomposeDimensionReducer(
                    vecs, dim, self.dim_reduction_meth, normalize=False)
                vecs = dim_reducer.reduced
                #dim_reducer.write()
                self.dim_reducers.append(dim_reducer)
                new_points = []
                for i, point in enumerate(points):
                    p = point.copy()
                    p.vec = vecs[i]
                    new_points.append(p)
                points = new_points
                if self.outlier_proportion is not None:
                    detect = OutlierDetector(
                        vecs, return_indicators=True,
                        proportion=self.outlier_proportion)
                    outliers = detect()
                    points = np.delete(new_points, outliers)
                    logger.debug(f'outlier prop={self.outlier_proportion}, ' +
                                 f'{len(outliers)} -> {len(points)}')
            pl = len(points)
            if pl > point_lim:
                vecs = np.stack(tuple(map(lambda p: p.vec, points)))
                sizes = tuple(map(map_size, points))
                end = start + pl
                embs.append((lab, color, vecs, points, sizes, start, end))
                start = end
        emb_vecs = np.concatenate(tuple(map(lambda x: x[2], embs)))
        if not self.calc_per_section:
            emb_vecs = self._dim_reduce(emb_vecs, dim)
        logger.debug(f'reduced embedding shape: {emb_vecs.shape}')
        outliers: np.ndarray = None
        if self.outlier_proportion is not None and not self.calc_per_section:
            detect = OutlierDetector(emb_vecs, return_indicators=True,
                                     proportion=self.outlier_proportion)
            outliers = detect()
        return embs, emb_vecs, colors, outliers

    def section_plot_3d(self):
        self._plotly_init()
        traces = []
        embs, emb_vecs, colors, outliers = self._plotly_embeds(3)

        if self.add_mean:
            mean = emb_vecs.mean(axis=0)
            lab = 'mean'
            trace = go.Scatter3d(
                text=[lab],
                x=[mean[0]],
                y=[mean[1]],
                z=[mean[2]],
                mode='markers',
                uid=lab,
                name=lab,
                marker={
                    'size': self.mean_point_size,
                    'opacity': 0.25,
                    'color': colors[0],
                },
            )
            traces.append(trace)

        for lab, color, vecs, points, sizes, start, end in embs:
            emb = emb_vecs[start:end]
            colors = color
            if outliers is not None:
                emb_outliers = outliers[start:end]
                if 0:
                    colors = np.where(emb_outliers, 'red', color)
                else:
                    emb = np.delete(emb, emb_outliers, axis=0)
                    sizes = np.delete(sizes, emb_outliers, axis=0)
                    points = np.delete(points, emb_outliers)
                    logger.debug(f'inliers: {emb.shape[0]}')
            trace = go.Scatter3d(
                text=tuple(map(str, points)),
                x=emb[:, 0],
                y=emb[:, 1],
                z=emb[:, 2],
                mode='markers',
                uid=lab,
                name=lab,
                marker={
                    'size': sizes,
                    'opacity': 0.6,
                    'color': colors,
                },
            )
            traces.append(trace)

        if isinstance(self.dim_reducers[0], DecomposeDimensionReducer):
            for di, dim_reducer in enumerate(self.dim_reducers):
                comps: Tuple[np.ndarray] = dim_reducer.get_components(emb_vecs)
                comps = tuple(it.islice(comps, self.plot_components))
                for n_comp, comp in enumerate(comps):
                    width = 7 - (n_comp*2)
                    start, end = comp.tolist()
                    lab = f'component {n_comp + 1}'
                    color = 'red' if len(self.dim_reducers) == 1 else embs[di][1]
                    #color = 'red'
                    trace = go.Scatter3d(
                        text=[lab],
                        x=[start[0], end[0]],
                        y=[start[1], end[1]],
                        z=[start[2], end[2]],
                        uid=lab,
                        name=lab,
                        marker={
                            'size': 3,
                            'opacity': .75,
                            #'color': 'red',
                            'color': color,
                        },
                        line={
                            'width': width,
                            #'color': color,
                            'color': 'red',
                        }
                    )
                    traces.append(trace)

        scene = {'showspikes': False}
        if self.axis_range is not None:
            scene['range'] = [-self.axis_range, self.axis_range]
        layout = go.Layout(
            title='Section Identification',
            #autosize=False,
            # width=self.width,
            height=self.height,
            scene={'xaxis': scene,
                   'yaxis': scene,
                   'zaxis': scene},
            margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
        )

        plot_figure = go.Figure(data=traces, layout=layout)
        plot_figure.update_layout(template='ggplot2')

        # render the plot
        plotly.offline.iplot(plot_figure)

    def section_plot_2d(self):
        self._plotly_init()
        traces = []
        embs, emb_vecs, colors, outliers = self._plotly_embeds(2)

        if self.add_mean:
            mean = emb_vecs.mean(axis=0)
            lab = 'mean'
            trace = go.Scatter(
                text=[lab],
                x=[mean[0]],
                y=[mean[1]],
                mode='markers',
                uid=lab,
                name=lab,
                marker={
                    'size': self.mean_point_size,
                    'opacity': 0.25,
                    'color': colors[0],
                },
            )
            traces.append(trace)

        for lab, color, vecs, points, sizes, start, end in embs:
            emb = emb_vecs[start:end]
            # configure the trace
            trace = go.Scatter(
                text=tuple(map(str, points)),
                x=emb[:, 0],
                y=emb[:, 1],
                mode='markers',
                uid=lab,
                name=lab,
                marker={
                    'size': sizes,
                    'opacity': 0.6,
                    'color': color,
                },
            )
            traces.append(trace)

        # configure the layout
        layout = go.Layout(
            title='Section Identification',
            # autosize=False,
            # width=self.width,
            height=self.height,
            margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
        )

        plot_figure = go.Figure(data=traces, layout=layout)
        plot_figure.update_layout(template='ggplot2')

        # render the plot
        plotly.offline.iplot(plot_figure)

    def section_plot_ax_2d(self, ax: plt.Axes = None):
        do_plot = ax is not None
        sec2points: Dict[str, List[CuiDataPoint]] = self._agg_doc_points()
        labs = sorted(sec2points.keys())
        colors = tuple(chain.from_iterable(
            [mcolors.BASE_COLORS.keys(), mcolors.TABLEAU_COLORS.keys()]))
        embs = []
        start = 0
        for lab, color in zip(labs, colors):
            points: List[np.ndarray] = sec2points[lab]
            pl = len(points)
            if pl > 1:
                vecs = np.stack(tuple(map(lambda p: p.vec, points)))
                end = start + pl
                embs.append((lab, color, vecs, start, end))
                start = end

        emb_vecs = np.concatenate(tuple(map(lambda x: x[2], embs)))
        emb_vecs = self._dim_reduce(emb_vecs, 2)

        for lab, color, vecs, start, end in embs:
            if do_plot:
                emb = emb_vecs[start:end]
                ax.scatter(emb[:, 0], emb[:, 1],
                           edgecolors='k', c=color, label=lab)
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels)
            else:
                print(lab, color, start, end)

    def clear(self):
        self._doc_points_pw.clear()
        self._agg_doc_points_pw.clear()
        self._agg_max_point_count_pw.clear()

    def _set_notebook_defaults(self):
        self.normalize = None
        self.outlier_significance = 0.3
        self.tfidf_threshold = 0.001

    def set_tfidf_section_by_index(self, ev_idx: int = None):
        df = pd.read_csv(self.expected_vars)
        labs = set(df.iloc[ev_idx]['columns'].split())
        ev = df.iloc[ev_idx]['ev']
        print(f'loading labels (ev={ev}): {labs}')
        self.tfidf_sections = set(labs)

    def _pca_var_analysis(self, sections: Set[str] = None, dim: int = 3) -> \
            DecomposeDimensionReducer:
        if sections is None:
            sections = {'past-medical-history', 'past-surgical-history'}
        self.tfidf_sections = sections
        sec2points: Dict[str, CuiDataPoint] = self._tfidf_trimmed()
        points: List[CuiDataPoint] = tuple(chain.from_iterable(
            map(lambda d: d.values(), sec2points.values())))
        vecs = np.stack(tuple(map(lambda p: p.vec, points)))
        dim_reducer = DecomposeDimensionReducer(
            vecs, dim, 'pca', normalize=self.normalize)
        return dim_reducer

    def _calc_metrics(self, dim: int = 3, secs: Set[str] = None):
        self._set_notebook_defaults()
        dim_reducer = self._pca_var_analysis(secs, dim)
        dim_reducer.write()

    def _write_ev_sheet(self, dim: int = 3, comb_size: int = 1,
                        keeps: Set[str] = None):
        self._set_notebook_defaults()
        out = False
        sec2points: Dict[str, CuiDataPoint] = self._tfidf_trimmed()
        all_secs = set(sec2points.keys())
        combs = map(set, it.combinations(all_secs, comb_size))
        if keeps is not None:
            combs = filter(lambda c: len(keeps & c) > 0, combs)
        combs = tuple(combs)
        print(f'combinations: {len(combs)}')
        rows = []
        for secs in combs:
            if out:
                print(f'sections: {secs}')
            try:
                reducer = self._pca_var_analysis(secs, dim)
                n_points = reducer.n_points
                pca = reducer.model
                ev = sum(list(pca.explained_variance_ratio_))
            except ValueError as e:
                print(e)
                continue
            evs = ', '.join(
                filter(lambda x: x != '0.00',
                       map(lambda x: f'{x:.2f}', pca.explained_variance_ratio_)))
            rows.append((n_points, ev, evs, ' '.join(sorted(secs))))
            if out: 
                print(f'explained variance: {ev}: {evs}')
                print('-' * 80)
        df = pd.DataFrame(rows, columns='points ev evs columns'.split())
        df = df.sort_values(['ev'], ascending=False)
        df.to_csv(self.expected_vars)
        print(f'wrote: {self.expected_vars}')

    def _trim_expected_vars(self):
        df = pd.read_csv(self.expected_vars)
        df = df[df['ev'] >= 0.6]
        df = df.sort_values(['points'], ascending=False)
        print(len(df))
        print(df)
        df.to_csv('/d/expected-vars.csv')
