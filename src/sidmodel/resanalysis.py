"""Results analysis.

"""
__author__ = 'Paul Landes'

from typing import Dict, Tuple
from dataclasses import dataclass, field
import sys
import logging
from io import TextIOBase
from pathlib import Path
import shutil
from tabulate import tabulate
import numpy as np
import pandas as pd
from zensols.config import Writable, ConfigFactory
from zensols.persist import persisted, PersistedWork, Stash
from zensols.nlp import FeatureToken, FeatureSentence, FeatureDocument
from zensols.deeplearn.model import LatexPerformanceMetricsDumper
from . import AnnotationNote, AnnotationStash

logger = logging.getLogger(__name__)


@dataclass
class ResultAnalyzer(Writable):
    config_factory: ConfigFactory
    results_dir: Path
    anon_stash: AnnotationStash
    feature_stash: Stash
    target_note_prop: Path
    ontology_file: Path
    temporary_dir: Path
    section_id_combination_limit: int = field(default=25)

    def __post_init__(self):
        self._tokens = PersistedWork(
            self.temporary_dir / 'tokens.dat', self,
            cache_global=True, mkdir=True)

    def clear(self):
        self._tokens.clear()

    @staticmethod
    def _prop_df(df: pd.DataFrame, col: str):
        df = df.groupby(col).agg({col: 'count'}).\
            rename(columns=({col: 'count'}))
        df = df.sort_values('count', ascending=False)
        tot = df['count'].sum()
        df['proportion'] = df['count'] / tot
        df[col] = df.index
        cols = df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        df = df[cols].reset_index(drop=True)
        return df

    @property
    def ontology(self) -> pd.DataFrame:
        return pd.read_csv(self.ontology_file)

    @property
    def ontology_summary(self) -> pd.DataFrame:
        df = self.ontology
        ont = pd.DataFrame(
            [[df['note_name'].drop_duplicates().count(),
              df['section_name'].drop_duplicates().count(),
              len(df)]],
            columns='notes sections relations'.split())
        return ont

    @property
    def note_description(self) -> pd.DataFrame:
        df = self.ontology
        df = df['note_name note_description'.split()].drop_duplicates()
        df = df.sort_values('note_name')
        df = df.rename(columns={
            'note_name': 'name', 'note_description': 'description'})
        return df

    @property
    def section_description(self) -> pd.DataFrame:
        df = self.ontology
        df = df['section_id section_name section_description'.split()].\
            drop_duplicates()
        df = df.sort_values('section_name')
        df = df.rename(columns={
            'section_name': 'name', 'section_description': 'description'})
        return df

    @property
    def notes(self) -> pd.DataFrame:
        def map_stash(an: AnnotationNote) -> Dict:
            dct = an.asdict()
            del dct['sections']
            return dct

        return pd.DataFrame(tuple(map(map_stash, self.anon_stash.values())))

    @property
    def annotations(self) -> pd.DataFrame:
        rows = []
        an: AnnotationNote
        for an in self.anon_stash.values():
            for sec in an.sections:
                rows.append((sec.id, an.category, len(sec.header_spans),
                             sec.body_span.begin, sec.body_span.end))
        df = pd.DataFrame(
            rows, columns='id category headers body_begin body_end'.split())
        return df

    @property
    @persisted('_tokens')
    def tokens(self) -> pd.DataFrame:
        rows = []
        doc: FeatureDocument
        for did, doc in self.feature_stash:
            sent: FeatureSentence
            for sent in doc:
                tok: FeatureToken
                for tok in sent:
                    rows.append((did, tok.sent_i, doc.category,
                                 tok.section_id_, tok.norm))
        return pd.DataFrame(rows, columns='doc sent category id text'.split())

    @property
    def corpus(self) -> pd.DataFrame:
        df = self.tokens
        return pd.DataFrame([
            ['documents', len(df['doc'].drop_duplicates())],
            ['annotations', len(self.annotations)],
            ['annotated_sentences', len(df[['doc', 'sent']].drop_duplicates())],
            ['total_tokens', len(df)],
            ['annotated_tokens', len(df[df['id'] != FeatureToken.NONE])],
        ], columns='description count'.split())

    @property
    def unique_sentence_section_id(self) -> pd.DataFrame:
        df = self.tokens
        # fold duplicate labels
        df = df['doc sent id'.split()].drop_duplicates()
        # per sentence unique labels
        df = df.groupby('doc sent'.split()).size().reset_index(name='count')
        return df

    @property
    def unique_sentence_section_id_counts(self) -> pd.DataFrame:
        df = self.unique_sentence_section_id.value_counts('count')
        df = pd.DataFrame({'count': df, 'unique_sections': df.index},
                          columns='unique_sections count'.split())
        df['proportion'] = df['count'] / df['count'].sum()
        df = df.reset_index(drop=True)
        return df

    @property
    def note_needs(self) -> pd.DataFrame:
        dfn = self.notes
        df = self._prop_df(dfn, 'category')
        dft = pd.read_csv(self.target_note_prop)
        df = df.merge(dft, on='category',
                      suffixes='_current _target'.split())
        df = df.rename(columns={'proportion_current': 'prop_current',
                                'proportion_target': 'prop_target'})
        df['prop_need'] = df['prop_target'] - df['prop_current']
        df['need'] = df['count'].sum() * df['prop_need']
        return df

    def _add_note_column(self, df: pd.DataFrame) -> pd.DataFrame:
        def map_id(s: str) -> str:
            names = dfo[dfo['section_id'] == s].sort_values('note_name')['note_name'].to_list()
            return ', '.join(names)

        dfo = self.ontology
        df['notes'] = df['id'].apply(map_id)
        return df

    def get_distributions(self) -> Tuple[str, str, pd.DataFrame]:
        capitalize = LatexPerformanceMetricsDumper.capitalize
        thous = LatexPerformanceMetricsDumper.format_thousand
        corp = self.corpus
        corp['description'] = corp['description'].apply(capitalize)
        corp['count'] = corp['count'].apply(lambda r: thous(r, apply_k=False))
        dists = [['corpus content', 'corpus-content', corp]]
        dists.extend([
            ['ontology count', 'ontology', self.ontology_summary],
            ['annotated note events',
             'note-events', self._prop_df(self.notes, 'category')],
            ['number of muliple unique annotations across sentences',
             'multi-sent-anons', self.unique_sentence_section_id_counts],
        ])
        df_ann = self._prop_df(self.annotations, 'category').merge(
            self._prop_df(self.tokens, 'category'), on='category',
            suffixes='_annotation _token'.split())
        df_sec_id = self._prop_df(self.annotations, 'id').merge(
            self._prop_df(self.tokens, 'id'), on='id',
            suffixes='_annotation _token'.split())
        df_sec_id = self._add_note_column(df_sec_id)
        df_sid_comb = df_sec_id.copy()
        df_sid_comb = df_sid_comb.sort_values('count_annotation', ascending=False)
        df_sid_comb['spans'] = df_sid_comb.apply(
            lambda r: f"{thous(r['count_annotation'])} ({r['proportion_annotation']*100:.0f}%)",
            axis=1)
        df_sid_comb['tokens'] = df_sid_comb.apply(
            lambda r: f"{thous(r['count_token'])} ({r['proportion_token']*100:.0f}%)",
            axis=1)
        df_sid_comb = df_sid_comb.drop(
            columns={'count_annotation', 'proportion_annotation',
                     'count_token', 'proportion_token'})
        df_sid_comb = df_sid_comb.rename(columns={'id': 'type'})
        df_sid_comb = df_sid_comb.head(self.section_id_combination_limit)
        cols = df_sid_comb.columns.to_list()
        cols[-1], cols[1] = cols[1], cols[-1]
        df_sid_comb = df_sid_comb[cols]
        dists.extend([
            ['annotation span and token categories',
             'annotation-category', df_ann],
            ['annotation span and token sections',
             'annotation-section-id', df_sec_id],
            ['annotation span and token sections combined',
             'annotation-section-id-combined', df_sid_comb],
            ['note descriptions', 'note-descriptions',
             self.note_description],
            ['section descriptions', 'section-descriptions',
             self.section_description],
            ])
        return dists

    def _format_df(self, name: str, df: pd.DataFrame,
                   table_name: bool = True, column_names: bool = True,
                   decimal: int = 2, make_percent: bool = True):
        def fmt_decimal(x: int) -> str:
            perc = ''
            if make_percent:
                x = x * 100
                perc = '%'
            fmt = f'{{:.{decimal}f}}{perc}'
            return fmt.format(x)

        capitalize = LatexPerformanceMetricsDumper.capitalize
        if table_name:
            name = capitalize(name)
        df = df.copy()
        df.columns = list(map(capitalize, df.columns))
        if decimal is not None:
            for cix, dtype in enumerate(df.dtypes):
                name = df.columns[cix]
                if dtype == np.float64:
                    df.loc[:, name] = df[name].apply(fmt_decimal)
        return name, df

    def _dump_dists(self, name: str, dists: Tuple[str, str, pd.DataFrame],
                    write_summary: bool):
        res_dir = self.results_dir / 'stats' / name
        if res_dir.is_dir():
            shutil.rmtree(res_dir)
        res_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = res_dir / 'manifest.txt'
        with open(manifest_path, 'w') as f:
            for desc, pref, df in dists:
                fname = f'{pref}.csv'
                out_path: Path = res_dir / fname
                df.to_csv(out_path, index=False)
                f.write(f'{fname}: {desc}\n')
                logger.info(f'wrote: {out_path}')
        logger.info(f'wrote: {manifest_path}')
        if write_summary:
            out_path = res_dir / 'summary.txt'
            with open(out_path, 'w') as f:
                self.write(writer=f)
            logger.info(f'wrote: {out_path}')

    def dump(self):
        fmts = []
        dists = self.get_distributions()
        for name, id, df in dists:
            name, df = self._format_df(name, df)
            fmts.append((name, id, df))
        self._dump_dists('raw', dists, True)
        self._dump_dists('formatted', fmts, False)

    def write_note_needs(self, depth: int = 0,
                         writer: TextIOBase = sys.stdout):
        df = self.note_needs
        self._write_line('note needed for annoatation', depth, writer)
        self._write_block(tabulate(df, df.columns, showindex=False),
                          depth, writer)

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        dists = self.get_distributions()
        for i, (desc, pref, df) in enumerate(dists):
            if i > 0:
                self._write_empty(writer)
            self._write_line(desc, depth, writer)
            self._write_block(tabulate(df, df.columns, showindex=False),
                              depth, writer)
        self._write_empty(writer)
        self.write_note_needs(depth, writer)

    def __call__(self):
        if 1:
            self.dump()
        else:
            self.write()
