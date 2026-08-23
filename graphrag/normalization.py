import re
import unicodedata


def normalize_evidence(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).lower()
    value = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", value)
    value = re.sub(r"[\s,;:，；：]+", "", value)
    return re.sub(r"(?<!\d)\.|\.(?!\d)", "", value)
