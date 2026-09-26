import streamlit as st
import duckdb
from ui_streamlit.components.pos_help import PTB_INFO, UPOS_INFO
from core.preprocessing.xml_parser import get_xml_attribute_columns
from core.modules.overview import get_pos_definitions

def apply_pending_tag_inserts(target_key):
    """
    Applies any queued tag string to st.session_state[target_key].
    Must be called BEFORE st.text_input(..., key=target_key) is instantiated to avoid StreamlitAPIException.
    """
    pending_key = f"_append_tag_{target_key}"
    # Check general pending key as well as specific key
    gen_pending_key = "_append_tag_any"
    tag_str = None
    if pending_key in st.session_state:
        tag_str = st.session_state.pop(pending_key)
    elif gen_pending_key in st.session_state:
        tag_str = st.session_state.get(gen_pending_key)
        # Keep gen_pending_key until all known inputs are updated or cleared on rerun
        
    if tag_str:
        current_val = st.session_state.get(target_key, "")
        if not current_val:
            # Fallback to shared kwic_search_term
            current_val = st.session_state.get('kwic_search_term', "")
            
        if current_val:
            if current_val.endswith(" ") or tag_str.startswith(" "):
                new_val = current_val + tag_str
            else:
                new_val = current_val + " " + tag_str
        else:
            new_val = tag_str
            
        st.session_state[target_key] = new_val
        st.session_state['kwic_search_term'] = new_val
        from ui_streamlit.state_manager import set_state
        set_state('kwic_search_term', new_val)

def _insert_tag_to_state(target_key, tag_str):
    """
    Queues a tag string for insertion into target_key and triggers a rerun.
    """
    st.session_state[f"_append_tag_{target_key}"] = tag_str
    st.session_state["_append_tag_any"] = tag_str
    st.rerun()

def _render_reference_pos_tags(target_key, filter_text=""):
    """
    Renders reference PTB and UPOS tag buttons.
    """
    categories = {
        "Verbs": ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "MD"],
        "Nouns": ["NN", "NNS", "NNP", "NNPS", "PRP", "PRP$"],
        "Adjectives": ["JJ", "JJR", "JJS"],
        "Adverbs": ["RB", "RBR", "RBS", "RP"],
        "Prepositions & Conjunctions": ["IN", "CC", "DT", "WDT", "WP", "WRB", "TO"],
        "Universal POS (UPOS)": list(UPOS_INFO.keys())
    }
    
    for cat_name, tag_list in categories.items():
        matching_tags = []
        for tag in tag_list:
            info = PTB_INFO.get(tag) or UPOS_INFO.get(tag, {})
            defn = info.get('defn', '')
            desc = info.get('desc', '')
            full_str = f"{tag} {defn} {desc}".lower()
            if not filter_text or filter_text in full_str:
                matching_tags.append((tag, defn))
        
        if matching_tags:
            st.caption(f"**{cat_name}**")
            cols = st.columns(4)
            for idx, (tag, defn) in enumerate(matching_tags):
                col_idx = idx % 4
                tag_token = f"_{tag}"
                with cols[col_idx]:
                    if st.button(tag_token, key=f"btn_pos_{tag}_{target_key}", help=defn, use_container_width=True):
                        _insert_tag_to_state(target_key, tag_token)

def render_tag_quick_insert(target_key, corpus_path=None, label="🏷️ Insert Tag"):
    """
    Renders a native Streamlit popover helper that lets users select and insert
    dynamically extracted POS tags (_TAG), XML structural tags (<tag segment="...">),
    Overview AI annotations (sentiment, topic, ent_type), or Dependency relations (dep_rel).
    """
    # Ensure any pending inserts are processed
    apply_pending_tag_inserts(target_key)

    with st.popover(label, help="Quickly insert Part-of-Speech (_), XML Metadata (<), Overview Annotations, or Dependency tags"):
        st.markdown("##### 🏷️ Tag Quick-Insert Helper")
        
        tab_pos, tab_dep, tab_xml = st.tabs(["POS Tags (_)", "Dependencies (dep_rel)", "XML & Overview Tags (<)"])
        
        with tab_pos:
            filter_text = st.text_input("Filter POS tags...", key=f"pos_search_filter_{target_key}", placeholder="e.g. verb, JJ, NN, noun").strip().lower()
            
            # DYNAMIC POS TAG EXTRACTION FROM ACTIVE CORPUS
            dynamic_tags = []
            pos_defs = {}
            
            if corpus_path:
                try:
                    pos_defs = get_pos_definitions(corpus_path) or {}
                    con = duckdb.connect(corpus_path, read_only=True)
                    cols_info = con.execute("PRAGMA table_info(corpus)").fetchall()
                    existing_cols = {c[1].lower() for c in cols_info}
                    if 'pos' in existing_cols:
                        raw_pos = con.execute("""
                            SELECT pos, count(pos) as cnt 
                            FROM corpus 
                            WHERE pos IS NOT NULL AND pos != '' AND pos != '##' 
                            GROUP BY pos 
                            ORDER BY cnt DESC 
                            LIMIT 100
                        """).fetchall()
                        dynamic_tags = [(r[0], r[1]) for r in raw_pos if r[0]]
                    con.close()
                except Exception:
                    dynamic_tags = []

            if corpus_path and dynamic_tags:
                st.caption("⚡ **Actual POS tags in loaded corpus (sorted by frequency):**")
                matching_dynamic = []
                for tag, count in dynamic_tags:
                    defn = pos_defs.get(tag) or PTB_INFO.get(tag, {}).get('defn') or UPOS_INFO.get(tag, {}).get('defn') or ""
                    desc = pos_defs.get(tag) or PTB_INFO.get(tag, {}).get('desc') or UPOS_INFO.get(tag, {}).get('desc') or ""
                    full_search = f"{tag} {defn} {desc}".lower()
                    if not filter_text or filter_text in full_search:
                        matching_dynamic.append((tag, count, defn))

                if matching_dynamic:
                    cols = st.columns(4)
                    for idx, (tag, count, defn) in enumerate(matching_dynamic):
                        col_idx = idx % 4
                        tag_token = f"_{tag}"
                        tooltip = f"{tag}: {defn} ({count:,} tokens)" if defn else f"{tag} ({count:,} tokens)"
                        with cols[col_idx]:
                            if st.button(tag_token, key=f"btn_dyn_pos_{tag}_{idx}_{target_key}", help=tooltip, use_container_width=True):
                                _insert_tag_to_state(target_key, tag_token)
                else:
                    st.info("No matching POS tags found for your search filter.")
            elif corpus_path:
                st.info("ℹ️ No POS tags detected in this corpus. This appears to be an untagged plain text corpus.")
                with st.expander("Reference POS Tags (PTB / UPOS)", expanded=False):
                    _render_reference_pos_tags(target_key, filter_text)
            else:
                # FALLBACK FOR WHEN NO CORPUS IS LOADED YET
                st.caption("ℹ️ *Reference POS tags (Load a corpus to see exact corpus tags):*")
                _render_reference_pos_tags(target_key, filter_text)

        with tab_dep:
            dep_tags = []
            has_dep = False
            
            if corpus_path:
                try:
                    con = duckdb.connect(corpus_path, read_only=True)
                    cols_info = con.execute("PRAGMA table_info(corpus)").fetchall()
                    existing_cols = {c[1].lower() for c in cols_info}
                    if 'dep_rel' in existing_cols:
                        raw_deps = con.execute("""
                            SELECT dep_rel, count(dep_rel) as cnt 
                            FROM corpus 
                            WHERE dep_rel IS NOT NULL AND dep_rel != '' 
                            GROUP BY dep_rel 
                            ORDER BY cnt DESC 
                            LIMIT 50
                        """).fetchall()
                        if raw_deps:
                            has_dep = True
                            dep_tags = [(r[0], r[1]) for r in raw_deps if r[0]]
                    con.close()
                except Exception:
                    pass

            std_dep_descriptions = {
                "nsubj": "nominal subject", "dobj": "direct object", "iobj": "indirect object",
                "csubj": "clausal subject", "ccomp": "clausal complement", "xcomp": "open clausal complement",
                "nmod": "nominal modifier", "appos": "appositional modifier", "nummod": "numeric modifier",
                "amod": "adjectival modifier", "advmod": "adverbial modifier", "acl": "clausal modifier of noun",
                "advcl": "adverbial clause modifier", "det": "determiner", "case": "case marking / preposition",
                "conj": "conjunct", "cc": "coordinating conjunction", "compound": "compound word",
                "aux": "auxiliary verb", "cop": "copula", "mark": "subordinator", "punct": "punctuation", "root": "sentence root"
            }

            if has_dep and dep_tags:
                st.caption("⚡ **Actual dependency relations in loaded corpus:**")
                dep_filter = st.text_input("Filter Dependency relations...", key=f"dep_search_filter_{target_key}", placeholder="e.g. nsubj, dobj, amod").strip().lower()
                
                matching_deps = []
                for dep_tag, count in dep_tags:
                    desc = std_dep_descriptions.get(dep_tag, f"dependency: {dep_tag}")
                    full_str = f"{dep_tag} {desc}".lower()
                    if not dep_filter or dep_filter in full_str:
                        matching_deps.append((dep_tag, count, desc))
                
                if matching_deps:
                    cols = st.columns(3)
                    for idx, (dep_tag, count, desc) in enumerate(matching_deps):
                        col_idx = idx % 3
                        dep_token = f"dep:{dep_tag}"
                        tooltip = f"{dep_tag}: {desc} ({count:,} occurrences)"
                        with cols[col_idx]:
                            if st.button(f"{dep_tag}", key=f"btn_dep_{dep_tag}_{idx}_{target_key}", help=tooltip, use_container_width=True):
                                _insert_tag_to_state(target_key, dep_token)
                else:
                    st.info("No matching dependency relations found in this corpus.")
            elif corpus_path:
                st.info("ℹ️ Dependency parsing has not been run on this corpus yet. You can run dependency parsing in Overview to generate syntactic relation tags.")
            else:
                st.info("ℹ️ Load a corpus with dependency parsing annotations to inspect available dependency tags.")

        with tab_xml:
            if not corpus_path:
                st.info("ℹ️ Load a corpus to see available XML structural and Overview annotation tags.")
            else:
                try:
                    con = duckdb.connect(corpus_path, read_only=True)
                    attr_cols = get_xml_attribute_columns(con)
                    
                    if not attr_cols:
                        st.info("ℹ️ No XML structural metadata or Overview annotations detected in this corpus.")
                    else:
                        xml_filter = st.text_input("Filter XML / Metadata tags...", key=f"xml_search_filter_{target_key}", placeholder="e.g. sentiment, topic, segment, sport").strip().lower()
                        
                        # Separate Overview AI annotations (sentiment, topic, ent_type) from structural XML attributes
                        ai_annotations = ['sentiment', 'topic', 'ent_type']
                        ai_cols = [c for c in attr_cols if c.lower() in ai_annotations]
                        meta_cols = [c for c in attr_cols if c.lower() not in ai_annotations]
                        
                        ordered_cols = []
                        if ai_cols:
                            ordered_cols.append(("⚡ Overview AI Annotations", ai_cols))
                        if meta_cols:
                            ordered_cols.append(("📂 XML Metadata & Structural Attributes", meta_cols))
                            
                        for group_label, cols_in_group in ordered_cols:
                            group_rendered = False
                            for attr in cols_in_group:
                                try:
                                    unique_vals_query = f'SELECT DISTINCT "{attr}" FROM corpus WHERE "{attr}" IS NOT NULL AND "{attr}" != \'\' LIMIT 25'
                                    raw_vals = [r[0] for r in con.execute(unique_vals_query).fetchall() if r[0] is not None]
                                    cleaned_vals = [str(v).strip() for v in raw_vals if str(v).strip() and str(v).lower() != 'nan']
                                    
                                    matching_vals = [v for v in cleaned_vals if not xml_filter or xml_filter in attr.lower() or xml_filter in v.lower()]
                                    
                                    if matching_vals:
                                        if not group_rendered:
                                            st.markdown(f"**{group_label}**")
                                            group_rendered = True
                                            
                                        st.caption(f"Attribute: `{attr}`")
                                        # Standalone tag format
                                        tag_generic = f'<{attr}>'
                                        if not xml_filter or xml_filter in tag_generic.lower():
                                            if st.button(f"Generic: {tag_generic}", key=f"btn_xml_gen_{attr}_{target_key}"):
                                                _insert_tag_to_state(target_key, tag_generic)
                                        
                                        cols = st.columns(2)
                                        for idx, val in enumerate(matching_vals):
                                            col_idx = idx % 2
                                            xml_token = f'<tag {attr}="{val}">'
                                            with cols[col_idx]:
                                                if st.button(xml_token, key=f"btn_xml_{attr}_{idx}_{target_key}", use_container_width=True):
                                                    _insert_tag_to_state(target_key, xml_token)
                                except Exception:
                                    pass
                    con.close()
                except Exception as e:
                    st.error(f"Error reading corpus metadata: {e}")
