import duckdb
import pandas as pd

def calculate_corpus_statistics(corpus_stats, db_path=None):
    """
    Calculates display metrics like Type/Token Ratio.
    Includes a self-healing check to query the DB if stats are missing or 0.
    """
    if not corpus_stats and not db_path:
        return {}
        
    c_stats = corpus_stats if corpus_stats else {}
    
    total_tokens = c_stats.get('total_tokens', 0)
    type_count = c_stats.get('unique_tokens', 0)
    
    if type_count == 0 and 'token_counts' in c_stats:
        type_count = len(c_stats['token_counts'])
    
    # Self-healing: if we have a DB but types/tokens are suspiciously low, re-calculate
    if db_path and (total_tokens == 0 or type_count == 0):
        try:
            con = duckdb.connect(db_path, read_only=True)
            res = con.execute("SELECT count(*), count(DISTINCT _token_low) FROM corpus").fetchone()
            total_tokens = res[0]
            type_count = res[1]
            con.close()
        except:
            pass

    ttr = (type_count / total_tokens) if total_tokens > 0 else 0
    
    return {
        'total_tokens': total_tokens,
        'unique_types': type_count,
        'ttr': round(ttr, 4)
    }

def get_restricted_stats(db_path, xml_where_clause="", xml_params=[]):
    """
    Calculates tokens, types and TTR for a specific XML restricted region.
    """
    if not db_path:
        return {}
    
    con = duckdb.connect(db_path, read_only=True)
    try:
        # total_tokens
        sql_total = f"SELECT count(*) FROM corpus WHERE 1=1 {xml_where_clause}"
        total_tokens = con.execute(sql_total, xml_params).fetchone()[0]
        
        # unique_types
        sql_types = f"SELECT count(DISTINCT _token_low) FROM corpus WHERE 1=1 {xml_where_clause}"
        unique_types = con.execute(sql_types, xml_params).fetchone()[0]
        
        ttr = (unique_types / total_tokens) if total_tokens > 0 else 0
        
        return {
            'total_tokens': total_tokens,
            'unique_types': unique_types,
            'ttr': round(ttr, 4)
        }
    finally:
        con.close()

def get_top_frequencies_v2(db_path, limit=100, xml_where_clause="", xml_params=[]):
    """
    Fetches the top frequency tokens with their POS (if available).
    """
    con = duckdb.connect(db_path, read_only=True)
    try:
        # Check if POS exists
        cols = [c[1] for c in con.execute("PRAGMA table_info(corpus)").fetchall()]
        has_pos = 'pos' in cols
        
        # Base query (exclude punctuation and purely numeric strings)
        filter_clause = "WHERE NOT regexp_matches(_token_low, '^[[:punct:]]+$') AND NOT regexp_matches(_token_low, '^[0-9]+$')"
        
        if xml_where_clause:
            filter_clause += xml_where_clause
            
        if has_pos:
            query = f"SELECT token, pos, count(*) as frequency FROM corpus {filter_clause} GROUP BY token, pos ORDER BY frequency DESC LIMIT {limit}"
        else:
            query = f"SELECT token, count(*) as frequency FROM corpus {filter_clause} GROUP BY token ORDER BY frequency DESC LIMIT {limit}"
            
        df = con.execute(query, xml_params).fetch_df()
        return df
    finally:
        con.close()

def get_unique_pos_tags(db_path, xml_where_clause="", xml_params=[]):
    """
    Fetches unique POS tags from the corpus, excluding dummy or empty tags.
    """
    con = duckdb.connect(db_path, read_only=True)
    try:
        cols = [c[1] for c in con.execute("PRAGMA table_info(corpus)").fetchall()]
        if 'pos' not in cols:
            return []
            
        query = "SELECT DISTINCT pos FROM corpus WHERE pos NOT IN ('##', '###', 'O', '', 'TAG') AND pos NOT LIKE '##%'"
        
        if xml_where_clause:
            query += xml_where_clause
            
        tags = [r[0] for r in con.execute(query, xml_params).fetchall()]
        return sorted(tags)
    finally:
        con.close()

def get_pos_definitions(db_path):
    """
    Fetches the POS definitions dictionary from the database.
    """
    if not db_path: return {}
    con = duckdb.connect(db_path, read_only=True)
    try:
        tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
        if 'pos_definitions' not in tables:
            return {}
        rows = con.execute("SELECT tag, definition FROM pos_definitions").fetchall()
        return {r[0]: r[1] for r in rows}
    except:
        return {}
    finally:
        con.close()

def save_pos_definitions(db_path, definitions):
    """
    Saves the POS definitions dictionary to the database.
    """
    if not db_path or not definitions: return False
    con = duckdb.connect(db_path)
    try:
        con.execute("CREATE TABLE IF NOT EXISTS pos_definitions (tag VARCHAR PRIMARY KEY, definition VARCHAR)")
        con.execute("DELETE FROM pos_definitions")
        data = [(k, v) for k, v in definitions.items() if v and v.strip()]
        if data:
            con.executemany("INSERT INTO pos_definitions VALUES (?, ?)", data)
        return True
    except Exception as e:
        print(f"Error saving definitions: {e}")
        return False
    finally:
        con.close()

def get_corpus_language(db_path):
    """Retrieves the corpus language from metadata table."""
    if not db_path: return "English"
    con = duckdb.connect(db_path)
    try:
        tables = con.execute("SHOW TABLES").fetchall()
        if ('corpus_metadata',) not in tables:
            return "English"
        res = con.execute("SELECT value FROM corpus_metadata WHERE key='language'").fetchone()
        return res[0] if res else "English"
    except:
        return "English"
    finally:
        con.close()

def set_corpus_language(db_path, language):
    """Saves the corpus language to metadata table."""
    if not db_path: return False
    con = duckdb.connect(db_path)
    try:
        con.execute("CREATE TABLE IF NOT EXISTS corpus_metadata (key VARCHAR PRIMARY KEY, value VARCHAR)")
        con.execute("INSERT INTO corpus_metadata VALUES ('language', ?) ON CONFLICT (key) DO UPDATE SET value=excluded.value", [language])
        return True
    except Exception as e:
        print(f"Error setting language: {e}")
        return False
    finally:
        con.close()
