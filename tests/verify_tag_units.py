import sys
import os
import re
import pandas as pd
import duckdb

sys.path.insert(0, 'c:\\Users\\priha\\Documents\\cortex')
from core.preprocessing.xml_parser import parse_xml_content_to_df
from core.preprocessing.cleaning import sanitize_xml_content

def test_query(title, query):
    print(f"\n--- {title}: '{query}' ---")
    content = """<?xml version="1.0" encoding="UTF-8"?>
<corpus name="XMLTagDemo" lang="EN">
<text id="1234">
<PN att="val">
John	NN	john
</PN>
works	VB	works
at	PP	at
<PN att="val">
New	JJ	new
Era	NN	era
Pizzeria	NN	pizzeria
</PN>
</text>
<text id="5678">
The	DT	the
<PN type="place">
Empire	NNP	Empire
State	NNP	State
Building	NNP	Building
</PN>
is	VBZ	be
in	IN	in
<PN type="place">
New	NNP	New
York	NNP	York
City	NNP	City
</PN>
.	.	.
</text>
</corpus>"""
    
    cleaned = sanitize_xml_content(content)
    result = parse_xml_content_to_df(cleaned, preserve_inline_tags=True)
    df = pd.DataFrame(result['df_data'])
    
    con = duckdb.connect(':memory:')
    con.execute('CREATE TABLE corpus AS SELECT * FROM df')
    con.execute('ALTER TABLE corpus ADD COLUMN id INTEGER')
    con.execute('CREATE SEQUENCE seq_id START 1')
    con.execute('UPDATE corpus SET id = nextval(\'seq_id\')')
    
    # Emulate concordance.py logic
    query_pattern = r'<[^>]+>|[^\s]+'
    search_terms = re.findall(query_pattern, query)
    
    def parse_term(term):
        match = re.match(r'<(\w+)(?:\s+(.+?))?>', term, re.IGNORECASE)
        if match: return {'type': 'xml_tag', 'tag': match.group(1).lower()}
        return {'type': 'word', 'val': term.lower()}
    
    comp = [parse_term(t) for t in search_terms]
    
    current_offset_exprs = []
    token_exprs = []
    for k, c in enumerate(comp):
        alias = f'c{k}'
        if c['type'] == 'xml_tag':
            tag_name = c['tag']
            current_offset_exprs.append(f"COALESCE({alias}.{tag_name}_len, 1)")
            token_exprs.append(f"(SELECT string_agg(token, ' ') FROM corpus c_sub WHERE c_sub.id BETWEEN {alias}.id AND {alias}.id + {alias}.{tag_name}_len - 1)")
        else:
            current_offset_exprs.append("1")
            token_exprs.append(f"{alias}.token")
    
    match_token_expr = "(" + " || ' ' || ".join(token_exprs) + ")" if len(token_exprs) > 1 else token_exprs[0]
    total_len_expr = " + ".join(current_offset_exprs)
    
    query_select = f"SELECT c0.id, {total_len_expr} as total_len, {match_token_expr} as match_token FROM corpus c0"
    query_joins = ""
    query_where = []
    
    for k, c in enumerate(comp):
        alias = f'c{k}'
        if k > 0:
            prev_alias = f"c{k-1}"
            prev_offset = current_offset_exprs[k-1]
            query_joins += f" JOIN corpus {alias} ON {alias}.id = {prev_alias}.id + {prev_offset} "
        
        if c['type'] == 'xml_tag':
            query_where.append(f"{alias}.in_{c['tag']}_start = TRUE")
        else:
            query_where.append(f"lower({alias}.token) = '{c['val']}'")
            
    final_query = query_select + query_joins + " WHERE " + " AND ".join(query_where)
    
    try:
        res = con.execute(final_query).fetch_df()
        print(f"Results found: {len(res)}")
        if not res.empty:
            print(res)
    except Exception as e:
        print(f"Error executing query: {e}")
        print(f"Query was: {final_query}")
    
    con.close()

test_query("Test 1: Single Tag", "<PN>")
test_query("Test 2: Multi-word search with Tag", "at <PN>")
test_query("Test 3: Tag followed by word", "<PN> is")
