"""Чтение неизменяемых исходных CSV и проверки фактической схемы."""
from .io import (KEY_COLUMNS, NUMERIC_COLUMNS, TRAIN_COLUMNS, TEST_COLUMNS,
                 read_dataset, validate_frame, summarize_frame, fingerprint)

__all__ = ['KEY_COLUMNS', 'NUMERIC_COLUMNS', 'TRAIN_COLUMNS', 'TEST_COLUMNS',
           'read_dataset', 'validate_frame', 'summarize_frame', 'fingerprint']
