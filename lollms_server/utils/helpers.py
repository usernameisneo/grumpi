# lollms_server/utils/helpers.py
import base64
import logging
import re
from typing import Union, Tuple, List, Dict, Any

logger = logging.getLogger(__name__)

def encode_base64(data: bytes) -> str:
    """Encodes bytes data into a Base64 string."""
    return base64.b64encode(data).decode('utf-8')

def decode_base64(encoded_str: str) -> bytes:
    """Decodes a Base64 string into bytes."""
    try:
        return base64.b64decode(encoded_str.encode('utf-8'))
    except base64.binascii.Error as e:
        logger.error(f"Error decoding base64 string: {e}")
        raise ValueError("Invalid Base64 string") from e

def extract_code_blocks(text: str, return_remaining_text: bool = False) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], str]]:
    """
    Extracts code blocks from text using string searching and slicing.
    Handles completeness and metadata tags.
    """
    code_blocks = []
    remaining_text_parts = []
    current_pos = 0
    block_index = 0

    while True:
        # Find the next opening delimiter
        start_delimiter_pos = text.find("```", current_pos)
        if start_delimiter_pos == -1:
            # No more opening delimiters found
            if return_remaining_text and current_pos < len(text):
                remaining_text_parts.append(text[current_pos:].strip())
            break

        # --- Store text before the block ---
        if return_remaining_text:
            pre_block_text = text[current_pos:start_delimiter_pos].strip()
            if pre_block_text:
                remaining_text_parts.append(pre_block_text)

        # Find the potential end of the first line (for language tag)
        content_start_pos = start_delimiter_pos + 3
        first_line_end = text.find('\n', content_start_pos)
        if first_line_end == -1:
            # No newline found after opening ```, tag must be on same line or not present
            first_line_end = len(text) # Consider rest of string as potential first line

        first_line = text[content_start_pos:first_line_end].strip()

        # Try to identify language tag
        lang_tag = 'unknown'
        potential_tag_match = re.fullmatch(r"[\w\-]+", first_line) # Check if the whole first line is a valid tag
        if potential_tag_match:
            lang_tag = first_line
            # Content actually starts after this line
            content_start_pos = first_line_end + 1
        # else: content starts right after ``` (content_start_pos remains as is)


        # Find the closing delimiter *after* the content start position
        end_delimiter_pos = text.find("```", content_start_pos)

        # --- Check preceding line for metadata ---
        file_name = ""
        section = ""
        preceding_text_segment = text[:start_delimiter_pos]
        preceding_lines = preceding_text_segment.strip().splitlines()
        if preceding_lines:
            last_line = preceding_lines[-1].strip()
            fn_match = re.match(r"<file_name>(.*?)</file_name>", last_line)
            if fn_match: file_name = fn_match.group(1).strip()
            elif last_line.startswith("## filename:"): file_name = last_line[len("## filename:"):].strip()
            sec_match = re.match(r"<section>(.*?)</section>", last_line)
            if sec_match: section = sec_match.group(1).strip()

        if end_delimiter_pos != -1:
            # Found a closing delimiter -> Complete block
            content = text[content_start_pos:end_delimiter_pos].strip()
            is_complete = True
            current_pos = end_delimiter_pos + 3 # Move past the closing ```
        else:
            # No closing delimiter found -> Incomplete block
            content = text[content_start_pos:].strip()
            is_complete = False
            current_pos = len(text) # Consumed the rest of the text

        block_info = {
            'index': block_index,
            'file_name': file_name,
            'section': section,
            'content': content,
            'type': lang_tag,
            'is_complete': is_complete
        }
        code_blocks.append(block_info)
        block_index += 1

        if not is_complete: # Stop searching if we hit an incomplete block
            break

    # --- Final return logic ---
    if return_remaining_text:
        text_without_blocks = '\n\n'.join(remaining_text_parts).strip()
        return code_blocks, text_without_blocks
    else:
        return code_blocks