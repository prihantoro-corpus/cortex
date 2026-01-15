import re

def strip_html(text):
    """Simple helper to remove HTML tags for LLM text processing."""
    if not isinstance(text, str):
        return str(text)
    return re.sub(r'<[^>]*>', '', text)

def sanitize_xml_content(xml_content):
    """
    Sanitizes XML content string for control characters, entities, 
    and structural issues.
    """
    if not xml_content:
        return ""
        
    # 1. Remove illegal control characters (keep \t, \n, \r)
    illegal_chars_re = re.compile(u'[^\u0020-\uD7FF\uE000-\uFFFD\u0009\u000A\u000D]', re.IGNORECASE)
    cleaned_xml_content = illegal_chars_re.sub('', xml_content)
    
    # 2. Fix unescaped ampersands (&) and broken entities
    cleaned_xml_content = re.sub(r'&(amp|lt|gt|quot|apos)(?![;])', r'&\1;', cleaned_xml_content)
    cleaned_xml_content = re.sub(r'&(?![A-Za-z0-9#]{2,5};|#)', r'&amp;', cleaned_xml_content)
    
    # 3. Selective Escaping for vertical format data
    if '\t' in cleaned_xml_content:
        lines = cleaned_xml_content.split('\n')
        sanitized_lines = []
        for line in lines:
            if '\t' in line:
                stripped = line.strip()
                if not (stripped.startswith('<') and '>' in stripped):
                    line = line.replace('<', '&lt;').replace('>', '&gt;')
            sanitized_lines.append(line)
        cleaned_xml_content = '\n'.join(sanitized_lines)

    # 4. Handle "Junk after document element" and Multiple roots
    # Remove redundant XML declarations if they exist in the middle
    cleaned_xml_content = re.sub(r'<\?xml[^?]*\?>', '', cleaned_xml_content)
    
    has_multiple_root_markers = False
    # Check for markers that suggest multiple documents or segments not wrapped in a single root
    for marker in ['<document', '<corpus', '<text', '<s ', '<s>']:
        if cleaned_xml_content.count(marker) > 1:
            has_multiple_root_markers = True
            break
            
    content_for_wrap_check = cleaned_xml_content.strip()
    
    # If starting with a closing tag or text, or has multiple roots, wrap it
    if has_multiple_root_markers or not content_for_wrap_check.startswith('<'):
        cleaned_xml_content = f"<root>\n{cleaned_xml_content}\n</root>"
    else:
        # Final cleanup for junk after the last expected root-level closing tag
        for root_tag in ['</document>', '</corpus>', '</text>', '</s>']:
            if root_tag in cleaned_xml_content:
                last_pos = cleaned_xml_content.rfind(root_tag)
                cleaned_xml_content = cleaned_xml_content[:last_pos + len(root_tag)]
                break

    return cleaned_xml_content.strip()
