
import xml.etree.ElementTree as ET
import pandas as pd
import re

# Mock XML with inline tags
xml_content = """
<corpus>
    <text id="001">
        <s n="1">
            <PN type="human" sex="male">John</PN> works at <PN type="place">New Era Pizzeria</PN>.
        </s>
    </text>
</corpus>
"""

def parse_inline_xml(xml_string):
    root = ET.fromstring(xml_string)
    
    tokens_data = []
    
    # Recursive function to traverse and track context
    def traverse(element, context_tags):
        # 1. Capture attributes of current tag (if relevant)
        current_context = context_tags.copy()
        tag_name = element.tag
        
        # Add boolean flag for tag presence
        current_context[f"in_{tag_name}"] = True
        
        # Add attributes prefixed by tag name
        for k, v in element.attrib.items():
            current_context[f"{tag_name}_{k}"] = v
            
        # 2. Process text BEFORE first child (the "text" attribute)
        if element.text:
            tokenize_and_add(element.text, current_context)
            
        # 3. Process children
        for child in element:
            traverse(child, current_context)
            # 4. Process tail text of child (text AFTER child tag but inside parent)
            if child.tail:
                tokenize_and_add(child.tail, current_context)

    def tokenize_and_add(text, context):
        if not text.strip(): return
        tokens = text.split() # Simple whitespace tokenizer for test
        for t in tokens:
            row = {'token': t}
            row.update(context)
            tokens_data.append(row)

    traverse(root, {})
    return pd.DataFrame(tokens_data)

print("--- Prototype Results ---")
df = parse_inline_xml(xml_content.strip())
print(df.to_string())
