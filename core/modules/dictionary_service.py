import duckdb
import pandas as pd
import re
from collections import Counter
import streamlit as st

# Common punctuation set matching typical corpus cleanup needs
PUNCTUATION = set(['.', ',', '!', '?', ';', ':', '(', ')', '[', ']', '"', "'", '“', '”', '‘', '’', '...', '-', '–', '—'])

def get_detailed_contextual_ngrams(corpus_db_path, query_word, xml_where_clause="", xml_params=[]):
    """
    Extracts specific bigrams and trigrams for the Dictionary module using DuckDB.
    """
    if not corpus_db_path or not query_word:
        return None
    
    try:
        con = duckdb.connect(corpus_db_path, read_only=True)
        q = query_word.lower()
        
        # Use Window Functions to get context in one pass (Context size = 3)
        sql = f"""
        SELECT 
            LAG(token, 3) OVER (ORDER BY id) as p3,
            LAG(token, 2) OVER (ORDER BY id) as p2,
            LAG(token, 1) OVER (ORDER BY id) as p1,
            token as node,
            LEAD(token, 1) OVER (ORDER BY id) as n1,
            LEAD(token, 2) OVER (ORDER BY id) as n2,
            LEAD(token, 3) OVER (ORDER BY id) as n3
        FROM corpus
        WHERE 1=1 {xml_where_clause}
        QUALIFY lower(node) = ?
        """
        
        df_res = con.execute(sql, xml_params + [q]).fetch_df()
        con.close()
        
        if df_res.empty: return None
        
        bg_l, bg_r = Counter(), Counter()
        tg_l, tg_c, tg_r = Counter(), Counter(), Counter()
        fg_l, fg_cl, fg_cr, fg_r = Counter(), Counter(), Counter(), Counter()
        
        def is_bad(t): return t is None or t.lower() in PUNCTUATION or t.isdigit()
        
        for _, row in df_res.iterrows():
            p3, p2, p1 = row['p3'], row['p2'], row['p1']
            node = row['node']
            n1, n2, n3 = row['n1'], row['n2'], row['n3']
            
            # Bigrams
            if not is_bad(n1): bg_l[(node, n1)] += 1
            if not is_bad(p1): bg_r[(p1, node)] += 1
            
            # Trigrams
            if not is_bad(n1) and not is_bad(n2): tg_l[(node, n1, n2)] += 1
            if not is_bad(p1) and not is_bad(n1): tg_c[(p1, node, n1)] += 1
            if not is_bad(p2) and not is_bad(p1): tg_r[(p2, p1, node)] += 1
            
            # Fourgrams
            if not is_bad(n1) and not is_bad(n2) and not is_bad(n3): fg_l[(node, n1, n2, n3)] += 1
            if not is_bad(p1) and not is_bad(n1) and not is_bad(n2): fg_cl[(p1, node, n1, n2)] += 1
            if not is_bad(p2) and not is_bad(p1) and not is_bad(n1): fg_cr[(p2, p1, node, n1)] += 1
            if not is_bad(p3) and not is_bad(p2) and not is_bad(p1): fg_r[(p3, p2, p1, node)] += 1
            
        return {
            'bigrams_left': bg_l.most_common(5),
            'bigrams_right': bg_r.most_common(5),
            'trigrams_left': tg_l.most_common(5),
            'trigrams_center': tg_c.most_common(5),
            'trigrams_right': tg_r.most_common(5),
            'fourgrams_left': fg_l.most_common(5),
            'fourgrams_center_left': fg_cl.most_common(5),
            'fourgrams_center_right': fg_cr.most_common(5),
            'fourgrams_right': fg_r.most_common(5)
        }
        
    except Exception as e:
        print(f"Error in get_detailed_contextual_ngrams: {e}")
        return None

def get_all_lemma_forms_details(corpus_db_path, target_word, xml_where_clause="", xml_params=[]):
    """
    Finds all word forms that share the same lemma(s) as the target word.
    Returns: (forms_df, unique_pos_list, unique_lemma_list)
    """
    if not corpus_db_path: return pd.DataFrame(), [], []
    term = target_word.lower()
    
    try:
        con = duckdb.connect(corpus_db_path, read_only=True)
        # XML-Aware Lemma discovery: find lemmas that are relevant to this word IN THIS REGION
        lemma_sql = f"SELECT DISTINCT lower(lemma) FROM corpus WHERE _token_low = ? AND lemma NOT LIKE '##%' {xml_where_clause}"
        lemma_list = [r[0] for r in con.execute(lemma_sql, [term] + xml_params).fetchall()]
        
        if not lemma_list:
            sql = f"SELECT lower(token) as token, pos, lemma, count(*) as freq FROM corpus WHERE _token_low = ? {xml_where_clause} GROUP BY 1,2,3"
            df = con.execute(sql, [term] + xml_params).fetch_df()
        else:
            placeholders = ','.join(['?']*len(lemma_list))
            sql = f"SELECT lower(token) as token, pos, lemma, count(*) as freq FROM corpus WHERE lower(lemma) IN ({placeholders}) {xml_where_clause} GROUP BY 1,2,3"
            df = con.execute(sql, lemma_list + xml_params).fetch_df()
            
        con.close()
        
        if df.empty: 
            return pd.DataFrame(columns=['Token', 'POS', 'Lemma']), [], []
        
        df.columns = ['Token', 'POS', 'Lemma', 'freq']
        
        # Filter unique_pos to only include tags for the SPECIFIC target word
        unique_pos = sorted(df[df['Token'] == term]['POS'].unique().tolist())
        # If for some reason the term itself isn't in the df (e.g. filtered by XML), fallback to all POS
        if not unique_pos:
            unique_pos = sorted(df['POS'].unique().tolist())
            
        unique_lemmas = sorted(df['Lemma'].unique().tolist())
        
        return df, unique_pos, unique_lemmas
        
    except Exception:
        return pd.DataFrame(), [], []

def get_related_forms_by_regex(corpus_db_path, target_word, xml_where_clause="", xml_params=[]):
    if not corpus_db_path: return []
    pat = f".*{re.escape(target_word)}.*"
    try:
        con = duckdb.connect(corpus_db_path, read_only=True)
        sql = f"SELECT DISTINCT token FROM corpus WHERE regexp_matches(_token_low, ?) {xml_where_clause} LIMIT 50"
        res = con.execute(sql, [pat] + xml_params).fetchall()
        con.close()
        return sorted([r[0] for r in res])
    except Exception:
        return []

def get_subcorpus_size(corpus_db_path, xml_where_clause="", xml_params=[]):
    if not corpus_db_path: return 0
    try:
        con = duckdb.connect(corpus_db_path, read_only=True)
        # Using count(*) on the restricted view
        sql = f"SELECT count(*) FROM corpus WHERE 1=1 {xml_where_clause}"
        count = con.execute(sql, xml_params).fetchone()[0]
        con.close()
        return count
    except Exception:
        return 0

def get_dictionary_examples(corpus_db_path, target_word, xml_where_clause="", xml_params=[]):
    if not corpus_db_path: return []
    try:
        con = duckdb.connect(corpus_db_path, read_only=True)
        term = target_word.lower()
        
        # 1. Introspect columns to find metadata
        # Standard columns that are NOT metadata
        standard_cols = {'id', 'token', 'pos', 'lemma', 'sent_id', '_token_low'}
        all_cols = [c[1] for c in con.execute("PRAGMA table_info(corpus)").fetchall()]
        
        # Metadata columns are anything else
        meta_cols = [c for c in all_cols if c not in standard_cols]
        
        # Build selection string for metadata
        # We need to select them from 'c' (the hit token row)
        meta_select_part = ""
        if meta_cols:
            meta_select_part = ", " + ", ".join([f"c.{c}" for c in meta_cols])
        
        # PRINCIPLE: Examples must contain the EXACT search word. 
        # We no longer expand to the whole lemma for example hits.
        where_sql = "c._token_low = ?"
        search_params = [term]

        # Prioritizing shorter sentences (< 15 tokens)
        hits_df = con.execute(f"""
            WITH sent_lengths AS (
                SELECT sent_id, count(*) as s_len
                FROM corpus
                GROUP BY sent_id
            )
            SELECT c.sent_id, c.pos, c.id as word_id, c.token as hit_token{meta_select_part}
            FROM corpus c
            JOIN sent_lengths s ON c.sent_id = s.sent_id
            WHERE {where_sql} {xml_where_clause}
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY c.pos 
                ORDER BY (s.s_len >= 15), s.s_len, random()
            ) = 1
        """, search_params + xml_params).fetch_df()
        
        examples = []
        for _, row in hits_df.iterrows():
            sent_id = row['sent_id']
            pos_tag = row['pos']
            word_id = row['word_id']
            
            # Extract metadata for this hit
            metadata = {}
            for mc in meta_cols:
                val = row[mc]
                if val is not None and str(val).strip() != "":
                     metadata[mc] = val
            
            sent_tokens = con.execute("SELECT token, id FROM corpus WHERE sent_id = ? ORDER BY id", [sent_id]).fetchall()
            
            target_idx = -1
            for idx, (t_text, t_id) in enumerate(sent_tokens):
                if t_id == word_id:
                    target_idx = idx
                    break
            
            if target_idx != -1:
                start_idx = max(0, target_idx - 7)
                end_idx = min(len(sent_tokens), target_idx + 8)
                truncated_tokens = sent_tokens[start_idx:end_idx]
                
                parts = []
                for t_text, t_id in truncated_tokens:
                    if t_id == word_id:
                        parts.append(f"<b>{t_text}</b>")
                    else:
                        parts.append(t_text)
                
                full_sent = " ".join(parts)
                full_sent = re.sub(r'\s+([.,!?;:])', r'\1', full_sent)
                if start_idx > 0: full_sent = "..." + full_sent
                if end_idx < len(sent_tokens): full_sent = full_sent + "..."
                
                # Return dict instead of tuple
                examples.append({
                    'pos': pos_tag,
                    'text': full_sent,
                    'metadata': metadata
                })
            
        con.close()
        return examples
    except Exception as e:
        print(f"Error in get_dictionary_examples: {e}")
        return []

def get_random_examples(corpus_db_path, target_word, limit=5, xml_where_clause="", xml_params=[]):
    if not corpus_db_path: return []
    try:
        con = duckdb.connect(corpus_db_path, read_only=True)
        hits = con.execute(f"SELECT DISTINCT sent_id FROM corpus WHERE _token_low = ? {xml_where_clause} ORDER BY random() LIMIT ?", [target_word.lower()] + xml_params + [limit]).fetchall()
        
        examples = []
        for (sent_id,) in hits:
            tokens = con.execute("SELECT token, _token_low FROM corpus WHERE sent_id = ? ORDER BY id", [sent_id]).fetchall()
            parts = []
            for t, tl in tokens:
                if tl == target_word.lower():
                    parts.append(f"<b>{t}</b>")
                else:
                    parts.append(t)
            examples.append(" ".join(parts))
        con.close()
        return examples
    except Exception:
        return []
