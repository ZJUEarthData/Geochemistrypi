import re
from functools import lru_cache
from typing import Union

# ---------- 1. 保护词库（避免误伤非化学列名） ----------
_SKIP_WORDS = frozenset({
    'sample', 'depth', 'id', 'date', 'time', 'lat', 'lon', 
    'latitude', 'longitude', 'station', 'location', 
    'unit', 'units', 'quality', 'flag', 'method', 'type',
    'description', 'comment', 'x', 'y', 'z', 
    'ppm', 'ppb', 'wt', 'vol', 'std', 'dev', 'mean'
})

# ---------- 2. 特殊基团修正 ----------
_SPECIAL_CASES = {
    'co': 'CO',
    'oh': 'OH',
    'no': 'NO',
}


@lru_cache(maxsize=1024)
def normalize_geochem_formula(text: str) -> str:
    """主函数：标准化地球化学元素符号"""
    if not isinstance(text, str):
        return str(text)

    text = text.strip()
    if not text:
        return text

    # 检查是否为纯保护词
    if text.lower() in _SKIP_WORDS:
        return text

    # 处理带分隔符的复合词
    parts = re.split(r'([_\s.])', text)
    if len(parts) > 1:
        normalized_parts = []
        for part in parts:
            if part in ('_', ' ', '.'):
                normalized_parts.append(part)
            else:
                normalized_parts.append(_normalize_single(part))
        return ''.join(normalized_parts)

    return _normalize_single(text)


def _normalize_single(text: str) -> str:
    """核心转换器：处理单个不含分隔符的字符串"""
    if not text or text.isdigit():
        return text

    lower_text = text.lower()

    if lower_text in _SKIP_WORDS:
        return text

    def _repl(match):
        letters = match.group(1)
        digits = match.group(2) or ''

        if letters in _SPECIAL_CASES:
            return _SPECIAL_CASES[letters] + digits

        if len(letters) == 1:
            return letters.upper() + digits
        elif len(letters) == 2:
            return letters.capitalize() + digits
        else:
            result = letters[0].upper()
            result += letters[1].lower()
            if len(letters) > 2:
                result += letters[2:].upper()
            return result + digits

    return re.sub(r'([a-z]+)(\d*)', _repl, lower_text)  # ← 注意这里多了第三个参数 text
