# lollms_server/utils/helpers.py
import base64
import ascii_colors as logging
import re # Ensure re is imported
from typing import Union, Tuple, List, Dict, Any, Optional
import base64 # Keep base64 functions if they were intended to be here
import binascii # Needed for base64 error handling

logger = logging.getLogger(__name__)

# --- Base64 Functions (Keep them here if they belong) ---
def encode_base64(data: bytes) -> str:
    """Encodes bytes data into a Base64 string."""
    return base64.b64encode(data).decode('utf-8')

def decode_base64(encoded_str: str) -> bytes:
    """Decodes a Base64 string into bytes."""
    try:
        # Remove potential data URI prefix if present
        if encoded_str.startswith('data:'):
            encoded_str = encoded_str.split(',', 1)[1]
        return base64.b64decode(encoded_str.encode('utf-8'), validate=True)
    except (binascii.Error, ValueError) as e: # Catch binascii.Error and ValueError
        logger.error(f"Error decoding base64 string: {e}")
        raise ValueError("Invalid Base64 string") from e

# --- Code Block Extraction (Keep as is) ---
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
        start_delimiter_pos = text.find("```", current_pos)
        if start_delimiter_pos == -1:
            if return_remaining_text and current_pos < len(text):
                remaining_text_parts.append(text[current_pos:].strip())
            break

        if return_remaining_text:
            pre_block_text = text[current_pos:start_delimiter_pos].strip()
            if pre_block_text:
                remaining_text_parts.append(pre_block_text)

        content_start_pos = start_delimiter_pos + 3
        first_line_end = text.find('\n', content_start_pos)
        if first_line_end == -1: first_line_end = len(text)
        first_line = text[content_start_pos:first_line_end].strip()

        lang_tag = 'unknown'
        potential_tag_match = re.fullmatch(r"[\w\-]+", first_line)
        if potential_tag_match:
            lang_tag = first_line
            content_start_pos = first_line_end + 1
        # else: content starts right after ```

        end_delimiter_pos = text.find("```", content_start_pos)

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
            content = text[content_start_pos:end_delimiter_pos].strip()
            is_complete = True
            current_pos = end_delimiter_pos + 3
        else:
            content = text[content_start_pos:].strip()
            is_complete = False
            current_pos = len(text)

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

        if not is_complete: break

    if return_remaining_text:
        text_without_blocks = '\n\n'.join(remaining_text_parts).strip()
        return code_blocks, text_without_blocks
    else:
        return code_blocks


# --- Updated Thought Tag Parsing ---
def parse_thought_tags(text: str) -> Tuple[str, Optional[str]]:
    """
    Parses text to extract content within <think>...</think> tags.
    If the text ends with an unclosed <think> tag, its content is included in thoughts.

    Args:
        text: The raw text potentially containing think tags.

    Returns:
        A tuple containing:
        - cleaned_text: The text with all <think> blocks (complete or incomplete at end) removed.
        - thoughts: The concatenated content of all found think blocks,
                    or None if no blocks were found. Incomplete blocks at the end
                    are marked.
    """
    cleaned_parts = []
    thoughts_parts = []
    last_end = 0
    incomplete_thought_marker = "--- Incomplete Thought Block ---"

    # 1. Process all COMPLETE <think>...</think> blocks first
    for match in re.finditer(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE): # Add IGNORECASE
        start, end = match.span()
        thought_content = match.group(1).strip()

        # Add text segment before this match to cleaned parts
        cleaned_parts.append(text[last_end:start])

        # Add the extracted thought content
        if thought_content:
            thoughts_parts.append(thought_content)

        # Update the position for the next search
        last_end = end

    # 2. Process the remaining text after the last complete </think> tag
    remaining_text = text[last_end:]

    # 3. Check for an unclosed <think> tag in the remaining text
    # Use case-insensitive search for the opening tag
    last_think_pos = -1
    think_tag_lower = "<think>"
    search_start = 0
    while True:
        pos = remaining_text.lower().find(think_tag_lower, search_start)
        if pos == -1:
            break
        last_think_pos = pos
        search_start = pos + 1 # Continue search after the found tag

    if last_think_pos != -1:
        # Found an opening <think> tag in the remaining part
        # Text *before* this last tag belongs to cleaned_text
        text_before_incomplete_think = remaining_text[:last_think_pos]
        cleaned_parts.append(text_before_incomplete_think)

        # Text *after* this last tag is the incomplete thought
        incomplete_thought_content = remaining_text[last_think_pos + len(think_tag_lower):].strip()

        if incomplete_thought_content:
            # Add a marker and the incomplete content to thoughts
            logger.warning(f"Found unclosed <think> tag at end. Capturing content: '{incomplete_thought_content[:100]}...'")
            thoughts_parts.append(incomplete_thought_marker)
            thoughts_parts.append(incomplete_thought_content)
        else:
             # The tag was the very last thing, no content after it
             logger.debug("Found unclosed <think> tag at the very end of the text.")
             thoughts_parts.append(incomplete_thought_marker) # Still add marker
             thoughts_parts.append("(Tag was empty or at end of text)")

    else:
        # No unclosed <think> tag found at the end, add all remaining text
        cleaned_parts.append(remaining_text)

    # 4. Combine the results
    cleaned_text = "".join(cleaned_parts).strip()
    thoughts = "\n\n".join(thoughts_parts).strip() if thoughts_parts else None

    if thoughts:
        logger.debug(f"Final parsed thoughts: {thoughts[:150]}...")

    return cleaned_text, thoughts

# --- Example Usage (for testing) ---
if __name__ == '__main__':
    incomplete_thought_marker = "--- Incomplete Thought Block ---"
    test_cases = [
        ("Simple text", ("Simple text", None)),
        ("Text with <think>thought 1</think> and <think>thought 2</think> end.", ("Text with  and  end.", "thought 1\n\nthought 2")),
        ("No closing tag <think>incomplete thought", ("No closing tag", f"{incomplete_thought_marker}\n\nincomplete thought")),
        ("Text <think>complete</think> then <think>incomplete", ("Text  then", f"complete\n\n{incomplete_thought_marker}\n\nincomplete")),
        ("<think>Complete</think>", ("", "Complete")),
        ("<think>incomplete only", ("", f"{incomplete_thought_marker}\n\nincomplete only")),
        ("Ends with tag<think>", ("Ends with tag", f"{incomplete_thought_marker}\n\n(Tag was empty or at end of text)")),
        ("Mixed case <ThiNk>Mixed case thought</tHink> end.", ("Mixed case  end.", "Mixed case thought")), # Requires IGNORECASE in regex
        ("Mixed case incomplete <THINK>Incomplete mixed", ("Mixed case incomplete", f"{incomplete_thought_marker}\n\nIncomplete mixed")), # Requires case-insensitive find
        ("", ("", None)),
        ("<think></think>", ("", None)), # Empty thoughts are ignored
    ]

    for i, (input_text, expected_output) in enumerate(test_cases):
        print(f"\n--- Test Case {i+1} ---")
        print(f"Input: '{input_text}'")
        cleaned, thought = parse_thought_tags(input_text)
        result = (cleaned, thought)
        print(f"Output: Cleaned='{cleaned}', Thoughts='{thought}'")
        print(f"Expected: Cleaned='{expected_output[0]}', Thoughts='{expected_output[1]}'")
        if result == expected_output:
            print("Result: PASS")
        else:
            print("Result: FAIL <<<<<<<<")