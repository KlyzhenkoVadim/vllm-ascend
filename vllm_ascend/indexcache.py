from enum import Enum


class IndexCacheDecodeMode(Enum):
    DISABLED = "disabled"
    ANCHOR = "anchor"
    REUSE = "reuse"
