import re
from typing import List, Optional

def extract_between(text: str, start_marker: str, end_marker: str) -> List[str]:
    """
    Extract content between start and end markers.
    Simplified version matching the provided source code.
    
    Args:
        text: The text to search in
        start_marker: The starting marker
        end_marker: The ending marker
        
    Returns:
        List of extracted content strings
    """
    pattern = re.escape(start_marker) + r"(.*?)" + re.escape(end_marker)
    return re.findall(pattern, text, flags=re.DOTALL)

def extract_from_templates(text: str, templates: List[str], regex: bool = False) -> List[str]:
    """
    Extract content based on templates with placeholders.
    
    Args:
        text: The text to search in
        templates: List of template strings with {} placeholders
        regex: Whether to treat templates as regex patterns
        
    Returns:
        List of extracted content strings
    """
    results = []
    
    for template in templates:
        if regex:
            # Use the template as is for regex matching
            matches = re.findall(template, text, re.DOTALL)
            results.extend(matches)
        else:
            # Convert template to regex pattern by escaping and replacing placeholders
            pattern = template.replace("{}", "(.*?)")
            pattern = re.escape(pattern).replace("\\(\\*\\*\\?\\)", "(.*?)")
            matches = re.findall(pattern, text, re.DOTALL)
            results.extend(matches)
    
    return results

def extract_sentences(text: str, max_sentences: Optional[int] = None) -> List[str]:
    """
    Extract sentences from text.
    
    Args:
        text: The text to extract sentences from
        max_sentences: Maximum number of sentences to extract
        
    Returns:
        List of sentences
    """
    if not text:
        return []
    
    # Simple sentence splitting (could be improved with NLP libraries)
    sentence_endings = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(sentence_endings, text)
    
    # Remove any empty strings
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if max_sentences:
        return sentences[:max_sentences]
    return sentences