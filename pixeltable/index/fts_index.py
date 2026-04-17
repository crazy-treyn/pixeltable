from __future__ import annotations

import sqlalchemy as sql
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex

import pixeltable.catalog as catalog
import pixeltable.exceptions as excs
import pixeltable.exprs as exprs
import pixeltable.type_system as ts
from pixeltable.env import Env

from .base import IndexBase


class FtsIndex(IndexBase):
    """
    Full-text search index using PostgreSQL GIN over to_tsvector(regconfig, text).
    """

    language: str

    def __init__(self, language: str = 'english', column: catalog.Column | None = None) -> None:
        self.language = language.strip().lower()
        if self.language == '':
            raise excs.Error('FTS index `language` must be a non-empty string')
        if column is not None and not column.col_type.is_string_type():
            raise excs.Error(
                f'Cannot create full-text search index on column {column.name!r}: '
                f'requires String type, got {column.col_type}'
            )

    def create_value_expr(self, c: catalog.Column) -> exprs.Expr:
        if not c.col_type.is_string_type():
            raise excs.Error(f'Type `{c.col_type}` of column {c.name!r} is not valid for a full-text search index.')
        return exprs.ColumnRef(c)

    def records_value_errors(self) -> bool:
        return False

    def get_index_sa_type(self, val_col_type: ts.ColumnType) -> sql.types.TypeEngine:
        return val_col_type.to_sa_type()

    def sa_create_stmt(self, store_index_name: str, sa_value_col: sql.Column) -> sql.Compiled:
        if Env.get().is_using_cockroachdb:
            raise excs.Error('Full-text search indexes are only supported on PostgreSQL')
        # Expression index: GIN over to_tsvector(language, text_col)
        tsvec = sql.func.to_tsvector(self.language, sa_value_col)
        sa_idx = sql.Index(store_index_name, tsvec, postgresql_using='gin')
        return CreateIndex(sa_idx, if_not_exists=True).compile(dialect=postgresql.dialect())

    def sa_drop_stmt(self, store_index_name: str, sa_value_col: sql.Column) -> sql.Compiled:
        # Drop by index name only (expression indexes are not tied to a simple column list).
        return sql.text(f'DROP INDEX IF EXISTS {store_index_name}').compile(dialect=postgresql.dialect())

    def match_clause(self, val_column: catalog.Column, query_literal: exprs.Literal) -> sql.ColumnElement:
        """SQL for to_tsvector(lang, col) @@ websearch_to_tsquery(lang, query)."""
        if not query_literal.col_type.is_string_type():
            raise excs.Error('search(): query must be a string')
        q = query_literal.val
        assert isinstance(q, str)
        tsvec = sql.func.to_tsvector(self.language, val_column.sa_col)
        tsq = sql.func.websearch_to_tsquery(self.language, q)
        return tsvec.op('@@')(tsq)

    def rank_clause(self, val_column: catalog.Column, query_literal: exprs.Literal) -> sql.ColumnElement:
        """SQL for ts_rank(to_tsvector(...), websearch_to_tsquery(...))."""
        if not query_literal.col_type.is_string_type():
            raise excs.Error('search(): query must be a string')
        q = query_literal.val
        assert isinstance(q, str)
        tsvec = sql.func.to_tsvector(self.language, val_column.sa_col)
        tsq = sql.func.websearch_to_tsquery(self.language, q)
        return sql.func.ts_rank(tsvec, tsq)

    @classmethod
    def display_name(cls) -> str:
        return 'fts'

    def as_dict(self) -> dict:
        return {'language': self.language}

    @classmethod
    def from_dict(cls, d: dict) -> FtsIndex:
        return cls(language=d.get('language', 'english'))
