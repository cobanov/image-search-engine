import lancedb
import pyarrow as pa


def get_lancedb_client(uri, token=None):
    if token is None:
        return lancedb.connect(uri)
    else:
        return lancedb.connect(uri, token=token)


def create_table(db, table_name, dim=2048):
    """Creates a table in the database if it does not exist. Raises an error if the table already exists."""
    if table_name in db.table_names():
        raise ValueError(f"Table '{table_name}' already exists. Creation aborted.")

    print(f"Creating table {table_name}")
    schema = pa.schema(
        [
            pa.field("vector", pa.list_(pa.float32(), dim)),
            pa.field("filepath", pa.string()),
        ]
    )

    return db.create_table(table_name, schema=schema)


def open_table(db, table_name):
    """Opens an existing table in the database. Raises an error if the table does not exist."""
    if table_name not in db.table_names():
        raise ValueError(f"Table '{table_name}' does not exist.")

    print(f"Opening table {table_name}")
    return db.open_table(table_name)


def add_data(table, data):
    """Adds data to the table."""
    table.append(pa.Table.from_pydict(data))


def get_search_results(table, query):
    return table.search(query)
