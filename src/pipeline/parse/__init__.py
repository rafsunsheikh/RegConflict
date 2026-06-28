"""Structural parsers. Dispatched by config.SOURCE_REGISTRY."""
from __future__ import annotations

from .base import Block, BaseParser
from .generic import GenericParser
from .mas_notices import MASNoticeParser
from .asic_rg import ASICRGParser
from .au_act import AUActParser
from .eu_regulation import EURegulationParser

_PARSERS: dict[str, BaseParser] = {
    "generic":        GenericParser(),
    "mas_notices":    MASNoticeParser(),
    "asic_rg":        ASICRGParser(),
    "au_act":         AUActParser(),
    "eu_regulation":  EURegulationParser(),
}


def get_parser(parser_id: str) -> BaseParser:
    return _PARSERS.get(parser_id, _PARSERS["generic"])


__all__ = ["Block", "BaseParser", "get_parser"]
