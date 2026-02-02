import xml.etree.ElementTree as ET
import re
import pandas as pd
try:
    from lxml import etree as LXML_ET
    HAS_LXML = True
except ImportError:
    HAS_LXML = False

from core.preprocessing.cleaning import sanitize_xml_content

def extract_xml_structure(xml_input, max_values=20):
    """
    Parses XML content and extracts structure.
    Returns (structure, error_message)
    """
    if xml_input is None:
        return None, None
        
    cleaned_xml_content = None
    if isinstance(xml_input, str):
        cleaned_xml_content = xml_input
    else:
        try:
            xml_input.seek(0)
            xml_content = xml_input.read().decode('utf-8', errors='ignore')
            cleaned_xml_content = sanitize_xml_content(xml_content)
        except Exception as e:
            return None, f"File Read Error: {e}"

    if not cleaned_xml_content:
        return None, "Empty content"

    try:
        if HAS_LXML:
            parser = LXML_ET.XMLParser(recover=True, encoding='utf-8')
            root = LXML_ET.fromstring(cleaned_xml_content.encode('utf-8'), parser=parser)
        else:
            root = ET.fromstring(cleaned_xml_content) 
    except Exception as e:
        return None, f"XML Parsing Error: {e}"

    structure = {}
    
    def process_element(element):
        tag = element.tag
        if tag not in structure:
            structure[tag] = {}
        
        for attr_name, attr_value in element.attrib.items():
            if attr_name not in structure[tag]:
                structure[tag][attr_name] = set()
            
            if len(structure[tag][attr_name]) < max_values:
                structure[tag][attr_name].add(attr_value)

        for child in element:
            process_element(child)

    process_element(root)
    return structure, None

def parse_xml_content_to_df(xml_input, force_vertical_xml=False, stanza_processor=None, lang_code='EN'):
    """
    Parses XML content, extracts sentences and IDs, and tokenizes/verticalizes.
    Returns dict with keys: lang_code, df_data, sent_map, attributes, error
    """
    cleaned_xml_content = None
    if isinstance(xml_input, str):
        cleaned_xml_content = xml_input
    else:
        try:
            xml_input.seek(0)
            xml_content = xml_input.read().decode('utf-8', errors='ignore')
            cleaned_xml_content = sanitize_xml_content(xml_content)
        except Exception as e:
            return {'error': f"Error reading XML file: {e}"}
    
    if not cleaned_xml_content:
        return {'error': "Empty content"}
    
    try:
        if HAS_LXML:
            parser = LXML_ET.XMLParser(recover=True, encoding='utf-8')
            root = LXML_ET.fromstring(cleaned_xml_content.encode('utf-8'), parser=parser)
        else:
            root = ET.fromstring(cleaned_xml_content)
            
        xml_lang = root.get('lang')
        if not xml_lang:
            lang_match = re.search(r'(<text\s+lang="([^"]+)">|<corpus\s+[^>]*lang="([^"]+)">)', cleaned_xml_content)
            if lang_match:
                xml_lang = lang_match.group(3) or lang_match.group(2)
        
        final_lang = xml_lang.upper() if xml_lang else lang_code.upper()
        if not final_lang: final_lang = 'XML'
            
    except Exception as e:
        return {'error': f"Tokenization Parse Error: {e}"}

    df_data = []
    sent_map = {}
    detected_attrs = {} 
    
    excluded_attrs = ('n', 'num', 'lang') # Removed 'id' from exclusion, manually handled below
    base_root_attrs = {}
    for k, v in root.attrib.items():
        if k.lower() in excluded_attrs: continue
        key_name = 'doc_id' if k.lower() == 'id' else k
        base_root_attrs[key_name] = v
    
    for k, v in base_root_attrs.items():
        if k not in detected_attrs: detected_attrs[k] = set()
        detected_attrs[k].add(v)

    elements_to_process = []
    pass1_tags = {'sent', 's', 'u', 'utterance'}
    pass2_tags = {'p', 'para', 'ab', 'div'} 
    pass3_tags = {'text'}

    def traverse_and_collect(element, current_attrs, target_tags):
        new_attrs = current_attrs.copy()
        
        # Prepare attributes for this element, checking exclusions and renaming id
        elem_attrs = {}
        for k, v in element.attrib.items():
            if k.lower() in excluded_attrs: continue
            key_name = 'doc_id' if k.lower() == 'id' else k
            elem_attrs[key_name] = v
            
        new_attrs.update(elem_attrs)
        
        if element.tag in target_tags:
            elements_to_process.append((element, new_attrs))
            return 
        for child in element:
            traverse_and_collect(child, new_attrs, target_tags)
            
    traverse_and_collect(root, {}, pass1_tags)
    if not elements_to_process:
        traverse_and_collect(root, {}, pass2_tags)
    if not elements_to_process:
         traverse_and_collect(root, {}, pass3_tags)

    if not elements_to_process:
        raw_sentence_text = "".join(root.itertext()).strip() 
        if raw_sentence_text:
            if stanza_processor:
                res = stanza_processor(raw_sentence_text, final_lang)
                if isinstance(res, tuple) and len(res) == 2:
                    stanza_records, err = res
                else:
                    stanza_records, err = res, None
                    
                if not err and stanza_records:
                    current_stanza_sent = -1
                    current_sent_text_parts = []
                    for rec in stanza_records:
                        if rec['sent_id'] != current_stanza_sent:
                            if current_stanza_sent != -1:
                                sent_map[sequential_id_counter] = " ".join(current_sent_text_parts)
                            
                            sequential_id_counter += 1
                            current_stanza_sent = rec['sent_id']
                            current_sent_text_parts = []
                        
                        row = {"token": rec['token'], "pos": rec['pos'], "lemma": rec['lemma'], "sent_id": sequential_id_counter}
                        row.update(base_root_attrs)
                        df_data.append(row)
                        current_sent_text_parts.append(rec['token'])
                    
                    if current_sent_text_parts:
                        sent_map[sequential_id_counter] = " ".join(current_sent_text_parts)
                        
                    return {'lang_code': final_lang, 'df_data': df_data, 'sent_map': sent_map, 'attributes': detected_attrs}

            cleaned_text = re.sub(r'([^\w\s])', r' \1 ', raw_sentence_text) 
            tokens = [t.strip() for t in cleaned_text.split() if t.strip()]
            if tokens:
               for token in tokens:
                    row = {"token": token, "pos": "TAG", "lemma": token, "sent_id": 1}
                    row.update(base_root_attrs)
                    df_data.append(row)
               sent_map[1] = raw_sentence_text
            return {'lang_code': final_lang, 'df_data': df_data, 'sent_map': sent_map, 'attributes': detected_attrs}
        return {'error': "No parseable content found"}

    sequential_id_counter = 0

    for sent_elem, combined_row_attrs in elements_to_process:
        for k, v in combined_row_attrs.items():
            if k not in detected_attrs: detected_attrs[k] = set()
            detected_attrs[k].add(v)

        sent_id_str = sent_elem.get('n') or sent_elem.get('id')
        sent_id = None
        if sent_id_str:
            try: sent_id = int(sent_id_str)
            except ValueError:
                sequential_id_counter += 1
                sent_id = sequential_id_counter
        else:
            sequential_id_counter += 1
            sent_id = sequential_id_counter

        word_tags = sent_elem.findall('.//w')
        raw_sentence_text = ""
        
        if word_tags:
            raw_tokens = []
            for w_elem in word_tags:
                token = w_elem.text.strip() if w_elem.text else ""
                if not token: continue
                pos = w_elem.get('pos') or w_elem.get('type') or "TAG"
                lemma = w_elem.get('lemma') or token
                row = {"token": token, "pos": pos, "lemma": lemma, "sent_id": sent_id}
                if combined_row_attrs: row.update(combined_row_attrs)
                df_data.append(row)
                raw_tokens.append(token)
            raw_sentence_text = " ".join(raw_tokens)
        else:
            raw_sentence_text = "".join(sent_elem.itertext()).strip() 
            inner_content = raw_sentence_text
            normalized_content = inner_content.replace('\r\n', '\n').replace('\r', '\n')
            lines = [line.strip() for line in normalized_content.split('\n') if line.strip()]
            
            is_vertical_format = False
            if lines:
                if force_vertical_xml: is_vertical_format = True
                else:
                    def is_line_vertical(l):
                        if '\t' in l: return True
                        words = re.split(r'\s+', l.strip())
                        return 1 <= len(words) <= 3 
                    is_vertical_format = sum(is_line_vertical(line) for line in lines) / len(lines) > 0.8
            
            if is_vertical_format:
                raw_tokens = []
                for line in lines:
                    parts = re.split(r'\t+', line.strip())
                    if not parts or not parts[0]: continue
                    token = parts[0]
                    pos = parts[1] if len(parts) > 1 else "TAG"
                    lemma = parts[2] if len(parts) > 2 else token
                    row = {"token": token, "pos": pos, "lemma": lemma, "sent_id": sent_id}
                    if combined_row_attrs: row.update(combined_row_attrs)
                    df_data.append(row)
                    raw_tokens.append(token)
            else:
                raw_text_to_tokenize = raw_sentence_text.replace('\n', ' ').replace('\t', ' ')
                if stanza_processor:
                    res = stanza_processor(raw_text_to_tokenize, final_lang)
                    if isinstance(res, tuple) and len(res) == 2:
                        stanza_records, err = res
                    else:
                        stanza_records, err = res, None
                    
                    if not err and stanza_records:
                        current_stanza_sent = -1
                        current_sent_text_parts = []
                        for rec in stanza_records:
                            if rec['sent_id'] != current_stanza_sent:
                                if current_stanza_sent != -1:
                                    sent_map[sequential_id_counter] = " ".join(current_sent_text_parts)
                                
                                sequential_id_counter += 1
                                current_stanza_sent = rec['sent_id']
                                current_sent_text_parts = []
                            
                            row = {"token": rec['token'], "pos": rec['pos'], "lemma": rec['lemma'], "sent_id": sequential_id_counter}
                            if combined_row_attrs: row.update(combined_row_attrs)
                            df_data.append(row)
                            current_sent_text_parts.append(rec['token'])
                        
                        if current_sent_text_parts:
                            sent_map[sequential_id_counter] = " ".join(current_sent_text_parts)
                        continue
                
                cleaned_text = re.sub(r'([^\w\s])', r' \1 ', raw_text_to_tokenize) 
                tokens = [t.strip() for t in cleaned_text.split() if t.strip()] 
                for token in tokens:
                    row = {"token": token, "pos": "TAG", "lemma": token, "sent_id": sent_id}
                    if combined_row_attrs: row.update(combined_row_attrs)
                    df_data.append(row)
        
        if raw_sentence_text:
            sent_map[sent_id] = raw_sentence_text.strip()
            
    if not df_data:
        return {'error': "No tokenized data extracted"}
        
    return {'lang_code': final_lang, 'df_data': df_data, 'sent_map': sent_map, 'attributes': detected_attrs}

def format_structure_data_hierarchical(structure_data, indent_level=0, max_values=20):
    """
    Formats the hierarchical XML structure data into an indented HTML list.
    """
    if not structure_data:
        return ""

    html_list = []
    
    def get_indent(level):
        return f'<span style="padding-left: {level * 1.5}em;">'

    for tag in sorted(structure_data.keys()):
        tag_data = structure_data[tag]
        tag_line = f'{get_indent(indent_level)}<span style="color: #6A5ACD; font-weight: bold;">&lt;{tag}&gt;</span></span><br>'
        html_list.append(tag_line)
        
        for attr in sorted(tag_data.keys()):
            values = sorted(list(tag_data.get(attr, set())))
            sampled_values_str = ", ".join(values[:max_values])
            if len(values) > max_values:
                sampled_values_str += f", ... ({len(values) - max_values} more unique)"

            attr_line = f'{get_indent(indent_level + 1)}'
            attr_line += f'<span style="color: #8B4513;">@{attr}</span> = '
            attr_line += f'<span style="color: #3CB371;">"{sampled_values_str}"</span></span><br>'
            html_list.append(attr_line)

    return "".join(html_list)

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
    """Checks if a column in the corpus table is purely integer-like (ignoring NULLs)."""
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
