# lollms_server/utils/helpers.py
import base64
import logging
import re # Import regex module
from typing import Union, Tuple, List, Dict, Any # Import Optional

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
# --- ORIGINAL LOLLMS EXTRACTOR FUNCTION ---
def extract_code_blocks(text: str, return_remaining_text: bool = False) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], str]]:
    """
    Extracts code blocks from text, using simplified completeness check.

    Identifies blocks delimited by ```. Extracts language tag, content,
    and determines if the block was closed. Optionally returns text
    with blocks removed. Also checks for <file_name> and <section> tags
    on the line preceding a block.

    Args:
        text (str): The input text.
        return_remaining_text (bool): If True, also return text without blocks.

    Returns:
        List[Dict] or Tuple[List[Dict], str]: See function docstring.
        Keys in dict: 'index', 'file_name', 'section', 'content', 'type', 'is_complete'.
    """
    remaining = text
    bloc_index = 0
    first_index = 0
    indices = [] # Stores absolute start/end positions of ``` delimiters

    # 1. Find all delimiter positions
    while True:
        try:
            index = remaining.index("```")
            indices.append(index + first_index)
            remaining = remaining[index + 3:]
            first_index += index + 3
            bloc_index += 1
        except ValueError:
            # If an odd number of delimiters were found, the last block is open.
            # Mark its end implicitly at the end of the original text.
            if bloc_index % 2 == 1:
                indices.append(len(text))
            break # No more ``` found

    code_blocks = []
    is_start_delimiter = True # Is the current index in 'indices' an opening delimiter?
    text_parts = [] # For reconstructing text without blocks
    last_processed_end = 0 # Track end of last segment (text or code block)

    # 2. Process delimiter pairs (or single opening delimiter for incomplete blocks)
    for i, delimiter_pos in enumerate(indices):
        if is_start_delimiter:
            # --- Initialize block info ---
            block_info = {
                'index': len(code_blocks), 'file_name': "", 'section': "",
                'content': "", 'type': "", 'is_complete': False
            }

            # --- Store text before the block ---
            if return_remaining_text:
                part = text[last_processed_end:delimiter_pos].strip()
                if part: text_parts.append(part)

            # --- Check preceding line for metadata ---
            preceding_text_segment = text[:delimiter_pos]
            preceding_lines = preceding_text_segment.strip().splitlines()
            if preceding_lines:
                last_line = preceding_lines[-1].strip()
                if last_line.startswith("<file_name>") and last_line.endswith("</file_name>"):
                    block_info['file_name'] = last_line[len("<file_name>"):-len("</file_name>")].strip()
                elif last_line.startswith("## filename:"):
                    block_info['file_name'] = last_line[len("## filename:"):].strip()
                if last_line.startswith("<section>") and last_line.endswith("</section>"):
                    block_info['section'] = last_line[len("<section>"):-len("</section>")].strip()

            # --- Extract language tag and content start position ---
            sub_text_start_pos = delimiter_pos + 3
            sub_text = text[sub_text_start_pos:]
            content_start_pos = 0 # Relative to sub_text start

            if len(sub_text) >= 0:
                try: find_space = sub_text.index(" ")
                except ValueError: find_space = float('inf')
                try: find_return = sub_text.index("\n")
                except ValueError: find_return = float('inf')

                lang_tag_end_pos = int(min(find_return, find_space))
                # Handle case where tag takes the whole first line
                if lang_tag_end_pos == float('inf'): lang_tag_end_pos = len(sub_text.split('\n', 1)[0]) # End of first line

                potential_tag = sub_text[:lang_tag_end_pos].strip()
                block_info["type"] = potential_tag if potential_tag else 'unknown'

                # Content starts after tag + newline/space, or right after tag if no space/newline
                if find_return == lang_tag_end_pos: content_start_pos = lang_tag_end_pos + 1
                elif find_space == lang_tag_end_pos: content_start_pos = lang_tag_end_pos + 1
                else: content_start_pos = lang_tag_end_pos
                content_start_pos = min(content_start_pos, len(sub_text)) # Ensure valid index

                # --- Determine content end and completeness ---
                if i + 1 < len(indices): # Check if a closing delimiter index exists
                    closing_delimiter_start_pos = indices[i + 1]
                    content_end_pos_in_subtext = closing_delimiter_start_pos - sub_text_start_pos

                    block_info["content"] = sub_text[content_start_pos:content_end_pos_in_subtext].strip()
                    block_info["is_complete"] = True # If closing index exists, it's complete

                    if return_remaining_text:
                        last_processed_end = closing_delimiter_start_pos + 3
                else:
                    # No closing delimiter index found - block runs to end of text
                    block_info["content"] = sub_text[content_start_pos:].strip()
                    block_info["is_complete"] = False
                    if return_remaining_text:
                        last_processed_end = len(text)

                code_blocks.append(block_info)

            # Next delimiter in 'indices' marks the end of this block
            is_start_delimiter = False
        else:
            # This delimiter marks the end of the previous block.
            # The next one (if any) will mark the start of a new block.
            is_start_delimiter = True

    # 3. Finalize return value
    if return_remaining_text:
        # Add any text remaining after the last processed segment
        if last_processed_end < len(text):
            part = text[last_processed_end:].strip()
            if part: text_parts.append(part)
        # Join non-code parts
        text_without_blocks = '\n\n'.join(text_parts)
        return code_blocks, text_without_blocks
    else:
        return code_blocks
# --- END CORRECTED AND SIMPLIFIED EXTRACTOR ---
# --- END ORIGINAL LOLLMS EXTRACTOR FUNCTION ---