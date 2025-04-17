# tests/utils/test_helpers.py
import pytest
from lollms_server.utils.helpers import extract_code_blocks

def test_extract_no_blocks():
    text = "This is plain text with no code blocks."
    blocks = extract_code_blocks(text)
    assert blocks == []

def test_extract_single_complete_block_no_lang():
    text = "Here is some code:\n```\nprint('Hello')\n```\nMore text."
    blocks = extract_code_blocks(text)
    assert len(blocks) == 1
    assert blocks[0]['content'] == "print('Hello')"
    assert blocks[0]['type'] == 'unknown' # Default if no language tag
    assert blocks[0]['is_complete'] == True

def test_extract_single_complete_block_with_lang():
    text = "```python\ndef main():\n    pass\n```"
    blocks = extract_code_blocks(text)
    assert len(blocks) == 1
    assert blocks[0]['content'] == "def main():\n    pass"
    assert blocks[0]['type'] == 'python'
    assert blocks[0]['is_complete'] == True

def test_extract_multiple_blocks():
    text = "First block:\n```bash\necho 'First'\n```\nSecond block:\n```python\n# Second\n```"
    blocks = extract_code_blocks(text)
    assert len(blocks) == 2
    assert blocks[0]['type'] == 'bash'
    assert blocks[0]['content'] == "echo 'First'"
    assert blocks[0]['is_complete'] == True
    assert blocks[1]['type'] == 'python'
    assert blocks[1]['content'] == "# Second"
    assert blocks[1]['is_complete'] == True

def test_extract_incomplete_block_at_end():
    text = "Starting code:\n```javascript\nfunction incomplete() {"
    blocks = extract_code_blocks(text)
    assert len(blocks) == 1
    assert blocks[0]['type'] == 'javascript'
    assert blocks[0]['content'] == "function incomplete() {"
    assert blocks[0]['is_complete'] == False # No closing ```

def test_extract_block_with_leading_trailing_whitespace():
    text = "  ```  sql \n  SELECT * FROM users;  \n```   "
    blocks = extract_code_blocks(text)
    assert len(blocks) == 1
    assert blocks[0]['type'] == 'sql'
    assert blocks[0]['content'] == "SELECT * FROM users;" # Content should be stripped
    assert blocks[0]['is_complete'] == True

def test_extract_empty_block():
    text = "Empty block: ``` ```"
    blocks = extract_code_blocks(text)
    assert len(blocks) == 1
    assert blocks[0]['type'] == 'unknown'
    assert blocks[0]['content'] == ""
    assert blocks[0]['is_complete'] == True

def test_extract_block_with_only_lang():
    text = "Lang only: ```python\n```"
    blocks = extract_code_blocks(text)
    assert len(blocks) == 1
    assert blocks[0]['type'] == 'python'
    assert blocks[0]['content'] == ""
    assert blocks[0]['is_complete'] == True

def test_extract_with_return_remaining_text():
    text = "Text before.\n```python\ncode\n```\nText after."
    blocks, remaining = extract_code_blocks(text, return_remaining_text=True)
    assert len(blocks) == 1
    assert blocks[0]['content'] == "code"
    # Check that remaining text doesn't include the block content or delimiters
    # The exact format might vary slightly based on stripping/joining logic in the helper
    assert "code" not in remaining
    assert "```" not in remaining
    assert "Text before." in remaining
    assert "Text after." in remaining
    print(f"Remaining: '{remaining}'") # Debug print
    # Adjust assertion based on actual output, e.g., joined with '\n\n'
    assert remaining.strip() == "Text before.\n\nText after." or remaining.strip() == "Text before.\nText after."


def test_extract_block_with_filename_tag():
    text = "Here's the config:\n<file_name>config.json</file_name>\n```json\n{\n  \"key\": \"value\"\n}\n```"
    blocks = extract_code_blocks(text)
    assert len(blocks) == 1
    assert blocks[0]['type'] == 'json'
    assert blocks[0]['file_name'] == 'config.json'
    assert blocks[0]['content'] == '{\n  "key": "value"\n}'
    assert blocks[0]['is_complete'] == True

def test_extract_block_with_alternative_filename_tag():
    text = "Here's the config:\n## filename: config.json\n```json\n{\n  \"key\": \"value\"\n}\n```"
    blocks = extract_code_blocks(text)
    assert len(blocks) == 1
    assert blocks[0]['type'] == 'json'
    assert blocks[0]['file_name'] == 'config.json'
    assert blocks[0]['content'] == '{\n  "key": "value"\n}'
    assert blocks[0]['is_complete'] == True

def test_extract_block_with_section_tag():
     text = "Update this part:\n<section>Update Function</section>\n```python\ndef update():\n    # TODO\n```"
     blocks = extract_code_blocks(text)
     assert len(blocks) == 1
     assert blocks[0]['type'] == 'python'
     assert blocks[0]['section'] == 'Update Function'
     assert blocks[0]['content'] == 'def update():\n    # TODO'
     assert blocks[0]['is_complete'] == True

def test_extract_adjacent_blocks():
    text = "```bash\ncmd1\n``````python\ncode1\n```"
    blocks = extract_code_blocks(text)
    assert len(blocks) == 2
    assert blocks[0]['type'] == 'bash'
    assert blocks[0]['content'] == "cmd1"
    assert blocks[0]['is_complete'] == True
    assert blocks[1]['type'] == 'python'
    assert blocks[1]['content'] == "code1"
    assert blocks[1]['is_complete'] == True