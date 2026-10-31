import re
from typing import List, Dict, Any

def clean_text(text: str) -> str:
    """Removes excessive whitespace while preserving markdown structure."""
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def split_into_sections(markdown_text: str) -> List[Dict[str, str]]:
    """Splits markdown by ## ITEM headers into distinct sections."""
    pattern = r'(## ITEM [0-9A-Z\.\s]+)'
    parts = re.split(pattern, markdown_text)
    
    sections = []
    current_header = "HEADER_INTRO"
    
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        if re.match(pattern, part):
            current_header = part.replace('#', '').strip()
        else:
            sections.append({
                "section": current_header,
                "content": part
            })
            
    return sections

def create_naive_chunks(
    sections: List[Dict[str, str]], 
    ticker: str, 
    chunk_size: int = 512, 
    overlap: int = 64
) -> List[Dict[str, Any]]:
    """
    Creates standard naive fixed-size chunks from sections with overlap.
    """
    chunks = []
    chunk_idx = 1
    
    for sec in sections:
        content = sec["content"]
        words = content.split()
        
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
            
            is_table = "|" in chunk_text and "---" in chunk_text
            
            chunk_id = f"{ticker}_{sec['section'].replace(' ', '_')[:15]}_CHUNK_{chunk_idx:03d}"
            chunks.append({
                "chunk_id": chunk_id,
                "ticker": ticker,
                "section": sec["section"],
                "content": chunk_text,
                "is_table": is_table,
                "word_count": len(chunk_words)
            })
            chunk_idx += 1
            
            if end == len(words):
                break
            start += (chunk_size - overlap)
            
    return chunks

def create_hierarchical_chunks(
    sections: List[Dict[str, str]], 
    ticker: str, 
    parent_size: int = 1024, 
    child_size: int = 128, 
    overlap: int = 32
) -> tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Hierarchical Chunking (Small-to-Big):
    Produces small child chunks (128 words) linked to parent documents (1024 words).
    Returns (child_chunks, parent_lookup_dict).
    """
    child_chunks = []
    parent_lookup = {}
    parent_idx = 1
    child_idx = 1
    
    for sec in sections:
        content = sec["content"]
        words = content.split()
        
        p_start = 0
        while p_start < len(words):
            p_end = min(p_start + parent_size, len(words))
            parent_words = words[p_start:p_end]
            parent_text = " ".join(parent_words)
            parent_id = f"{ticker}_{sec['section'].replace(' ', '_')[:15]}_PARENT_{parent_idx:03d}"
            
            parent_lookup[parent_id] = parent_text
            parent_idx += 1
            
            # Split this parent into small child chunks for precise vector indexing
            c_start = 0
            while c_start < len(parent_words):
                c_end = min(c_start + child_size, len(parent_words))
                c_words = parent_words[c_start:c_end]
                c_text = " ".join(c_words)
                
                child_id = f"{parent_id}_CHILD_{child_idx:04d}"
                child_chunks.append({
                    "chunk_id": child_id,
                    "parent_id": parent_id,
                    "ticker": ticker,
                    "section": sec["section"],
                    "content": c_text,
                    "is_table": "|" in c_text and "---" in c_text,
                    "word_count": len(c_words)
                })
                child_idx += 1
                
                if c_end == len(parent_words):
                    break
                c_start += (child_size - overlap)
                
            if p_end == len(words):
                break
            p_start += (parent_size - (overlap * 2))
            
    return child_chunks, parent_lookup
