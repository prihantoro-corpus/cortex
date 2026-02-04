import duckdb
import pandas as pd
import re
import sys
import os

# Exactly replicate concordance.py's parse_term and logic
def parse_term(term, is_raw_mode=False):
    xml_tag_match = re.match(r'<(\w+)(?:\s+(.+?))?>', term, re.IGNORECASE)
    if xml_tag_match:
        tag_name = xml_tag_match.group(1).lower()
        attrs_str = xml_tag_match.group(2)
        attrs = {}
        if attrs_str:
            attr_pattern = r'(\w+)=(["\'])([^"\']*)\2'
            for match in re.finditer(attr_pattern, attrs_str):
                attrs[match.group(1).lower()] = match.group(3)
        return {'type': 'xml_tag', 'tag': tag_name, 'attrs': attrs}
    return {'type': 'word', 'val': term.lower()}

def run_test(query):
    CORPORA_DIR = 'c:\\Users\\priha\\Documents\\cortex\\corpora'
    demo_rel_path = 'english/xml_tag_demo.xml'
    
    sys.path.insert(0, 'c:\\Users\\priha\\Documents\\cortex')
    from core.preprocessing.xml_parser import parse_xml_content_to_df
    from core.preprocessing.cleaning import sanitize_xml_content

    with open(os.path.join(CORPORA_DIR, demo_rel_path), 'r', encoding='utf-8') as f:
        content = f.read()

    cleaned = sanitize_xml_content(content)
    result = parse_xml_content_to_df(cleaned, preserve_inline_tags=True)
    df = pd.DataFrame(result['df_data'])
    
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE corpus AS SELECT * FROM df")
    # Add ID column
    con.execute("ALTER TABLE corpus ADD COLUMN id INTEGER")
    con.execute("CREATE SEQUENCE seq_id START 1")
    con.execute("UPDATE corpus SET id = nextval('seq_id')")
    
    # Replication of generate_kwic logic
    query_pattern = r'<[^>]+>|[^\s]+'
    search_terms = re.findall(query_pattern, query)
    search_components = [parse_term(term) for term in search_terms]
    
    query_select = f"SELECT c0.id, c0.token as match_token FROM corpus c0"
    query_joins = ""
    query_where = []
    query_params = []
    
    for k, comp in enumerate(search_components):
        alias = f"c{k}"
        if k > 0: query_joins += f" JOIN corpus {alias} ON {alias}.id = c0.id + {k} "
        
        if comp['type'] == 'xml_tag':
            tag_name = comp['tag']
            attrs = comp['attrs']
            tag_col = f"in_{tag_name}"
            query_where.append(f"{alias}.{tag_col} = TRUE")
            for attr_key, attr_val in attrs.items():
                attr_col = f"{tag_name}_{attr_key}"
                query_where.append(f"{alias}.{attr_col} = ?")
                query_params.append(attr_val)
        elif comp['type'] == 'word':
            query_where.append(f"lower({alias}.token) = ?")
            query_params.append(comp['val'])

    final_query = query_select + query_joins
    if query_where: final_query += " WHERE " + " AND ".join(query_where)
    
    print(f"Query: {query}")
    print(f"Executing: {final_query} with {query_params}")
    try:
        res = con.execute(final_query, query_params).fetch_df()
        print(f"Results: {len(res)}")
        if not res.empty:
            print(res.head(5))
    except Exception as e:
        print(f"ERROR: {e}")
    
    con.close()

print("--- Test 1: Single Tag <PN> ---")
run_test("<PN>")
print("\n--- Test 2: Tag with attribute <PN type=\"person\"> ---")
run_test("<PN type=\"person\">")
print("\n--- Test 3: Multi-word search 'at <PN>' ---")
run_test("at <PN>")
print("\n--- Test 4: Mixed Case <pn> ---")
run_test("<pn>")


print("--- Test 1: Single Tag <PN> ---")
run_test("<PN>")
print("\n--- Test 2: Tag with attribute <PN type=\"person\"> ---")
run_test("<PN type=\"person\">")
print("\n--- Test 3: Multi-word search 'at <PN>' ---")
# Note: 'at' might be lowercase in DB if using _token_low
run_test("at <PN>")
