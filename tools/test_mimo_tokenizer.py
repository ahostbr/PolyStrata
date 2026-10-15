"""Compare the GGUF tokenizer with Xiaomi's pinned tokenizer.json; weights are not read.

python tools/test_mimo_tokenizer.py --gguf MODEL.gguf --reference tokenizer.json
Requires tokenizers, regex and jinja2. A complete GGUF header alone also suffices.
"""
import argparse
import itertools
from pathlib import Path
import sys
import tempfile
import unicodedata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from strata_tokenizer import Tokenizer, extract
from gguf_reader import GGUFFile
from serve.frontend import ChatTemplate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    from tokenizers import Tokenizer as Reference
    reference = Reference.from_file(str(args.reference))
    actual = Tokenizer.from_gguf(args.gguf)
    assert actual.pre == "qwen2", actual.pre
    corpus = ["Hello, MiMo!", "你好，世界", "1234567890", "a\r\nb\n\n  ", "🙂👨‍👩‍👧‍👦", "café e\u0301"]
    corpus += [a+b+c for a, b, c in itertools.product(
        ["a", " x", "can't", "中文", "1", "_", "", "\n"],
        ["\u0301", "\u0308", "\u0327", "a", "_j", "42", " ", "\t"],
        ["_j", "abc", "!", "\n", "123", "", " 🐈", "é"])]
    metadata = GGUFFile(args.gguf).metadata
    with tempfile.TemporaryDirectory() as directory:
        extract(args.gguf, directory)
        restored = Tokenizer.from_dir(Path(directory) / "tokenizer")
        template_path = Path(directory) / "chat_template.jinja"
        template_path.write_text(metadata["tokenizer.chat_template"], encoding="utf-8")
        template = ChatTemplate(template_path)
        for thinking in (False, True):
            prompt = template.render([{"role": "user", "content": "Say hello."}], enable_thinking=thinking)
            assert "Say hello." in prompt and "assistant" in prompt
            corpus.append(prompt)
    for text in corpus:
        expected = reference.encode(text, add_special_tokens=False).ids
        got = actual.encode(text, parse_special=True)
        assert got == expected, (repr(text), got, expected)
        assert restored.encode(text, parse_special=True) == expected, repr(text)
        assert actual.decode(got) == unicodedata.normalize("NFC", text), repr(text)
    print(f"PASS MiMo tokenizer: {len(corpus)} cases match Xiaomi token IDs and NFC round-trip; extracted tokenizer and chat template work")


if __name__ == "__main__":
    main()
