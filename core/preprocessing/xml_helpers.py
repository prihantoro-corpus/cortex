def get_xml_attribute_columns(con):
    """Identifies columns in the DuckDB corpus table that are XML segment-level attributes."""
    try:
        cols_info = con.execute("PRAGMA table_info(corpus)").fetchall()
        db_cols = [c[1] for c in cols_info]
        standard_cols = {'token', 'pos', 'lemma', 'sent_id', '_token_low', 'id', 'filename'}
        return [col for col in db_cols if col not in standard_cols]
    except:
        return []

def is_integer_col(con, col_name):
    """
    Checks if a column in the corpus table is purely integer-like (ignoring NULLs).
    """
    try:
        sql = f"SELECT count(*) FROM corpus WHERE {col_name} IS NOT NULL AND TRY_CAST({col_name} AS BIGINT) IS NULL"
        fail_count = con.execute(sql).fetchone()[0]
        return fail_count == 0
    except:
        return False

def apply_xml_restrictions(filters):
    """
    Returns a SQL WHERE clause fragment based on user-selected XML attribute filters.
    Returns (sql_fragment, params_list)
    """
    if not filters:
        return "", []
    
    clauses = []
    params = []
    for attr, val_data in filters.items():
        if val_data['type'] == 'list':
            vals = val_data['values']
            placeholders = ', '.join(['?'] * len(vals))
            clauses.append(f"{attr} IN ({placeholders})")
            params.extend(vals)
        elif val_data['type'] == 'range':
            min_v = val_data['min']
            max_v = val_data['max']
            clauses.append(f"TRY_CAST({attr} AS BIGINT) BETWEEN ? AND ?")
            params.extend([min_v, max_v])
            
    return " AND " + " AND ".join(clauses), params
