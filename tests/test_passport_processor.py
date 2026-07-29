"""Tests for passport MRZ helpers and the passport response contract."""

import io

import cv2
import numpy as np

from src.document_ocr.passport import processor
from src.document_ocr.passport.processor import (
    _build_full_name,
    _parse_names_from_line1,
    extract_mrz_from_image,
)

SAMPLE_MRZ = (
    "P<NGANWOKOMA<<EMMANUEL<UGWUNNA<<<<<<<<<<<<<<\n"
    "B016136813NGA9805275M280116491859727756<<<84"
)


class _StubEngine:
    """OCR engine stub that returns fixed MRZ text for any image variant."""

    def __init__(self, text: str):
        self._text = text

    def is_available(self) -> bool:
        return True

    def read_text_from_image(self, image):
        return [self._text]

    def group_boxes_into_lines(self, boxes) -> str:
        return boxes[0]


def _flat_image_stream():
    """Build a plain mid-gray JPEG that passes the glare/scan quality checks."""
    image = np.full((1000, 750, 3), 128, dtype=np.uint8)
    return io.BytesIO(cv2.imencode(".jpg", image)[1].tobytes())


def test_full_name_joins_given_names_before_surname():
    """Passport full names should read given names first, like the NIN slip output."""
    assert _build_full_name("EMMANUEL UGWUNNA", "NWOKOMA") == "EMMANUEL UGWUNNA NWOKOMA"


def test_full_name_falls_back_to_whichever_name_is_present():
    """A weak MRZ can leave one name empty, so the other should still be returned."""
    assert _build_full_name("", "NWOKOMA") == "NWOKOMA"
    assert _build_full_name("EMMANUEL", "") == "EMMANUEL"


def test_full_name_is_empty_when_no_names_were_read():
    """Blank or missing names should produce an empty string, not stray whitespace."""
    assert _build_full_name("", "") == ""
    assert _build_full_name("   ", None) == ""


def test_full_name_uses_mrz_line1_names():
    """The full name should match what TD3 line 1 encodes for a real passport."""
    surname, given_names = _parse_names_from_line1(SAMPLE_MRZ.split("\n")[0])

    assert _build_full_name(given_names, surname) == "EMMANUEL UGWUNNA NWOKOMA"


def test_passport_response_exposes_full_name_and_document_type():
    """Consumers key off `full_name` and a top-level `document_type`, like the NIN route."""
    original_engine = processor.engine
    processor.engine = _StubEngine(SAMPLE_MRZ)
    try:
        result = extract_mrz_from_image(_flat_image_stream())
    finally:
        processor.engine = original_engine

    assert result["success"] is True
    assert result["document_type"] == "PASSPORT"
    assert result["data"]["surname"] == "NWOKOMA"
    assert result["data"]["given_names"] == "EMMANUEL UGWUNNA"
    assert result["data"]["full_name"] == "EMMANUEL UGWUNNA NWOKOMA"
    assert result["data"]["passport_number"] == "B01613681"
    assert result["data"]["nin"] == "91859727756"
    assert result["verification"]["document_type"] == "P"
