"""Tests for PostgreSQL full-text search (FTS) indexes."""

from __future__ import annotations

import pytest

import pixeltable as pxt
from pixeltable.env import Env
from pixeltable.index import FtsIndex

from .utils import list_store_indexes, reload_catalog, validate_update_status


def _skip_if_cockroach() -> None:
    if Env.get().is_using_cockroachdb:
        pytest.skip('Full-text search indexes require PostgreSQL')


@pytest.mark.usefixtures('uses_db')
class TestFtsIndex:
    def test_add_drop_fts_index(self) -> None:
        _skip_if_cockroach()
        t = pxt.create_table('fts_add_drop', {'text': pxt.String}, if_exists='replace')
        t.insert([{'text': 'hello world'}])
        t.add_fts_index('text', idx_name='text_fts')
        idx_info = t._tbl_version.get().idxs_by_name['text_fts']
        assert isinstance(idx_info.idx, FtsIndex)
        assert idx_info.idx.language == 'english'
        store_idx_name = t._tbl_version.get()._store_idx_name(idx_info.id)
        assert store_idx_name in list_store_indexes(t)

        t.drop_fts_index(idx_name='text_fts')
        assert 'text_fts' not in t._tbl_version.get().idxs_by_name
        assert store_idx_name not in list_store_indexes(t)

    def test_fts_rejects_non_string_column(self) -> None:
        _skip_if_cockroach()
        t = pxt.create_table('fts_bad_col', {'n': pxt.Int}, if_exists='replace')
        with pytest.raises(pxt.Error, match='String'):
            t.add_fts_index('n')

    def test_search_filter_and_stemming(self) -> None:
        _skip_if_cockroach()
        t = pxt.create_table('fts_search_basic', {'text': pxt.String}, if_exists='replace')
        rows = [
            'The quick brown fox jumps over the lazy dog.',
            'A fast brown fox leaped over the sleepy dog.',
            'Pixeltable is a declarative AI data infrastructure.',
            'The cat in the hat.',
            'PostgreSQL full text search is powerful and fast.',
            'The quick brown fox.',
        ]
        validate_update_status(t.insert([{'text': s} for s in rows]), expected_rows=len(rows))
        t.add_fts_index('text')

        fox_rows = t.where(t.text.search('fox')).select(t.text).collect()
        assert len(fox_rows) == 3

        jump_rows = t.where(t.text.search('jumping')).select(t.text).collect()
        assert len(jump_rows) >= 1
        assert any('jumps' in r['text'] for r in jump_rows)

    def test_search_rank_order(self) -> None:
        _skip_if_cockroach()
        t = pxt.create_table('fts_rank', {'text': pxt.String}, if_exists='replace')
        rows = [
            'The quick brown fox jumps over the lazy dog.',
            'A fast brown fox leaped over the sleepy dog.',
            'Pixeltable is a declarative AI data infrastructure.',
            'The cat in the hat.',
            'PostgreSQL full text search is powerful and fast.',
            'The quick brown fox.',
        ]
        validate_update_status(t.insert([{'text': s} for s in rows]), expected_rows=len(rows))
        t.add_fts_index('text')

        q = t.text.search('quick brown fox')
        res = t.select(t.text, rank=q.rank).where(q).order_by(q.rank, asc=False).collect()
        assert len(res) >= 2
        ranks = [r['rank'] for r in res]
        assert ranks == sorted(ranks, reverse=True)
        texts = {r['text'] for r in res}
        assert 'The quick brown fox.' in texts
        assert 'The quick brown fox jumps over the lazy dog.' in texts

    def test_websearch_or_query(self) -> None:
        _skip_if_cockroach()
        t = pxt.create_table('fts_or', {'text': pxt.String}, if_exists='replace')
        validate_update_status(
            t.insert(
                [
                    {'text': 'only fast'},
                    {'text': 'only quick'},
                ]
            ),
            expected_rows=2,
        )
        t.add_fts_index('text')
        res = t.where(t.text.search('quick OR fast')).select(t.text).collect()
        assert len(res) == 2

    def test_french_language_index(self) -> None:
        _skip_if_cockroach()
        t = pxt.create_table('fts_fr', {'text': pxt.String}, if_exists='replace')
        t.insert([{'text': 'Les chats sont dans le jardin.'}])
        t.add_fts_index('text', language='french')
        fts_infos = [i for i in t._tbl_version.get().idxs_by_name.values() if isinstance(i.idx, FtsIndex)]
        assert len(fts_infos) == 1
        assert fts_infos[0].idx.language == 'french'
        res = t.where(t.text.search('chat')).select(t.text).collect()
        assert len(res) == 1

    def test_update_reflects_in_search(self) -> None:
        _skip_if_cockroach()
        t = pxt.create_table('fts_update', {'id': pxt.Int, 'text': pxt.String}, if_exists='replace')
        validate_update_status(t.insert([{'id': 1, 'text': 'original content'}]), expected_rows=1)
        t.add_fts_index('text')
        assert len(t.where(t.text.search('original')).collect()) == 1
        assert len(t.where(t.text.search('replaced')).collect()) == 0

        validate_update_status(t.update({'text': 'replaced content'}, where=t.id == 1))
        assert len(t.where(t.text.search('original')).collect()) == 0
        assert len(t.where(t.text.search('replaced')).collect()) == 1

    def test_reload_catalog(self) -> None:
        _skip_if_cockroach()
        t = pxt.create_table('fts_reload', {'text': pxt.String}, if_exists='replace')
        t.insert([{'text': 'hello world'}])
        t.add_fts_index('text', idx_name='ridx')
        q = t.select(t.text, m=t.text.search('hello')).where(t.text.search('hello'))
        assert len(q.collect()) == 1

        reload_catalog()
        t2 = pxt.get_table('fts_reload')
        assert len(t2.select(t2.text, m=t2.text.search('hello')).where(t2.text.search('hello')).collect()) == 1

    def test_drop_by_column_requires_single_fts(self) -> None:
        _skip_if_cockroach()
        t = pxt.create_table('fts_multi', {'text': pxt.String}, if_exists='replace')
        t.insert([{'text': 'x'}])
        t.add_fts_index('text', idx_name='a')
        t.add_fts_index('text', idx_name='b')
        with pytest.raises(pxt.Error, match='multiple indices'):
            t.drop_fts_index(column='text')

    @pytest.mark.cockroachdb
    def test_fts_raises_on_cockroach(self) -> None:
        if not Env.get().is_using_cockroachdb:
            pytest.skip('Requires CockroachDB')
        t = pxt.create_table('fts_crdb', {'text': pxt.String}, if_exists='replace')
        with pytest.raises(pxt.Error, match='PostgreSQL'):
            t.add_fts_index('text')
