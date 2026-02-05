from typing import List


def simple_tokenize(text: str) -> List[str]:
    # A minimal tokenizer for mixed Japanese/English/code.
    text = text.lower()
    for ch in [
        "\n",
        "\r",
        "\t",
        "　",
        ",",
        ".",
        "。",
        "、",
        "?",
        "！",
        "!",
        "：",
        ":",
        "（",
        "）",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        '"',
        "'",
        "・",
        "／",
        "/",
        "\\",
        "－",
        "-",
        "〜",
        "~",
    ]:
        text = text.replace(ch, " ")
    tokens = [t for t in text.split(" ") if t]
    return tokens
