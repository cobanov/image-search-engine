import os

import lancedb
import pyarrow as pa


def get_lancedb_client(db_path):
    """Get or create a LanceDB connection"""
    os.makedirs(db_path, exist_ok=True)
    return lancedb.connect(db_path)


def create_table(db, table_name, dim=512):
    """Create a new table with the specified schema"""
    schema = pa.schema(
        [
            pa.field("vector", pa.list_(pa.float32(), dim)),
            pa.field("filepath", pa.string()),
            pa.field("id", pa.string()),
            pa.field("scientific_name", pa.string(), nullable=True),
            pa.field("common_name", pa.string(), nullable=True),
        ]
    )

    return db.create_table(table_name, schema=schema, mode="overwrite")


def create_research_table(db, table_name, dim=512):
    """Create a new table with the specified schema"""
    schema = pa.schema(
        [
            pa.field("vector", pa.list_(pa.float32(), dim)),
            pa.field("scientific_name", pa.string()),
            pa.field("common_name", pa.string()),
            pa.field("result", pa.string()),
        ]
    )

    return db.create_table(table_name, schema=schema, mode="overwrite")

def open_table(db, table_name):
    """Open an existing table"""
    return db.open_table(table_name)


def add_data(table, data):
    """Adds data to the table."""
    table.append(pa.Table.from_pydict(data))


def get_search_results(table, query):
    return table.search(query)
