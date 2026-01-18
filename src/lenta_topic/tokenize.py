import re
from lenta_topic.config import *


def tokenize(text: str) -> list[str]:
    return re.findall(CYR_TOKEN_PATTERN, text)
