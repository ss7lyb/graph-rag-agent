from typing import List, Dict, Any
from collections import defaultdict
import logging

def num_tokens_from_string(text: str) -> int:
    """Estimate the number of tokens in a text string"""
    # Simple heuristic: approximately 4 characters per token for English text
    return len(text) // 4

def kb_prompt(kbinfos: Dict[str, List[Dict[str, Any]]], max_tokens: int = 4096) -> List[str]:
    """
    Format knowledge base information into a structured prompt, following the original implementation.
    
    Args:
        kbinfos: Dictionary containing chunks and document aggregations
        max_tokens: Maximum tokens for the resulting prompt
        
    Returns:
        List of formatted information blocks
    """
    # Extract content_with_weight from chunks
    knowledges = []
    for ck in kbinfos.get("chunks", []):
        content = ck.get("content_with_weight", ck.get("text", ""))
        if content:
            knowledges.append(content)
    
    # Limit total tokens
    used_token_count = 0
    chunks_num = 0
    for i, c in enumerate(knowledges):
        used_token_count += num_tokens_from_string(c)
        chunks_num += 1
        if max_tokens * 0.97 < used_token_count:
            knowledges = knowledges[:i]
            logging.warning(f"Not all the retrieval into prompt: {i+1}/{len(knowledges)}")
            break
    
    # Get document information
    doc_aggs = kbinfos.get("doc_aggs", [])
    docs = {d.get("doc_id", ""): d for d in doc_aggs}
    
    # Group chunks by document
    doc2chunks = defaultdict(lambda: {"chunks": [], "meta": {}})
    for i, ck in enumerate(kbinfos.get("chunks", [])[:chunks_num]):
        # Get document name or ID
        doc_id = ck.get("doc_id", ck.get("chunk_id", "unknown").split("_")[0] if "_" in ck.get("chunk_id", "") else "unknown")
        doc_name = doc_id
        
        # Add URL if available
        url_prefix = f"URL: {ck['url']}\n" if "url" in ck else ""
        
        # Get content
        content = ck.get("content_with_weight", ck.get("text", ""))
        
        # Add chunk to document group
        doc2chunks[doc_name]["chunks"].append(f"{url_prefix}ID: {i}\n{content}")
        
        # Add metadata if available
        if doc_id in docs:
            doc2chunks[doc_name]["meta"] = {
                "title": docs[doc_id].get("title", doc_id),
                "type": docs[doc_id].get("type", "unknown")
            }
    
    # Format final knowledge blocks
    formatted_knowledges = []
    for doc_name, cks_meta in doc2chunks.items():
        txt = f"\nDocument: {doc_name} \n"
        
        # Add metadata
        for k, v in cks_meta["meta"].items():
            txt += f"{k}: {v}\n"
            
        txt += "Relevant fragments as following:\n"
        
        # Add chunk content
        for chunk in cks_meta["chunks"]:
            txt += f"{chunk}\n"
            
        formatted_knowledges.append(txt)
    
    # If no chunks were found
    if not formatted_knowledges:
        return ["No relevant information found in the knowledge base."]
        
    return formatted_knowledges