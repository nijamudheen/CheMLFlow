from __future__ import annotations

import logging
import os
from typing import Iterable

import pandas as pd
from pydantic import ValidationError

from .models import ContractSpec

logger = logging.getLogger(__name__)


def _warn_or_raise(message: str, warn_only: bool) -> None:
    if warn_only:
        logger.warning(message)
    else:
        raise ValueError(message)

def _format_hint(contract: ContractSpec) -> str:
    if contract.description:
        return f" Hint: {contract.description}"
    return ""


def _columns_present(columns: Iterable[str], required: Iterable[str]) -> list[str]:
    missing = [col for col in required if col not in columns]
    return missing


def bind_output_path(contract: ContractSpec, path: str) -> ContractSpec:
    return contract.model_copy(update={"output_path": path})


def validate_contract(contract: ContractSpec, warn_only: bool = True) -> bool:
    if not contract.output_path:
        _warn_or_raise(
            f"[{contract.name}] Missing output_path on contract; cannot validate.{_format_hint(contract)}",
            warn_only,
        )
        return False
    if contract.output_kind == "dir":
        return validate_dir(contract.output_path, contract, warn_only=warn_only)
    if contract.output_kind == "file":
        return validate_file(contract.output_path, contract, warn_only=warn_only)
    return validate_csv(contract.output_path, contract, warn_only=warn_only)


def validate_dir(path: str, contract: ContractSpec, warn_only: bool = True) -> bool:
    if not os.path.exists(path):
        _warn_or_raise(
            f"[{contract.name}] Missing directory: {path}.{_format_hint(contract)}",
            warn_only,
        )
        return False
    if not os.path.isdir(path):
        _warn_or_raise(
            f"[{contract.name}] Expected directory: {path}.{_format_hint(contract)}",
            warn_only,
        )
        return False
    return True


def validate_file(path: str, contract: ContractSpec, warn_only: bool = True) -> bool:
    if not os.path.exists(path):
        _warn_or_raise(
            f"[{contract.name}] Missing file: {path}.{_format_hint(contract)}",
            warn_only,
        )
        return False
    if not os.path.isfile(path):
        _warn_or_raise(
            f"[{contract.name}] Expected file: {path}.{_format_hint(contract)}",
            warn_only,
        )
        return False
    return True


def validate_csv(path: str, contract: ContractSpec, warn_only: bool = True) -> bool:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        _warn_or_raise(
            f"[{contract.name}] Missing file: {path}.{_format_hint(contract)}",
            warn_only,
        )
        return False
    except Exception as exc:  # pragma: no cover - unexpected read failures
        _warn_or_raise(
            f"[{contract.name}] Failed to read CSV {path}: {exc}.{_format_hint(contract)}",
            warn_only,
        )
        return False

    columns = list(df.columns)
    ok = True

    if contract.min_columns and len(columns) < contract.min_columns:
        _warn_or_raise(
            f"[{contract.name}] Expected at least {contract.min_columns} column(s), got {len(columns)}.{_format_hint(contract)}",
            warn_only,
        )
        ok = False

    missing_required = _columns_present(columns, contract.required_columns)
    if missing_required:
        _warn_or_raise(
            f"[{contract.name}] Missing required columns: {missing_required}.{_format_hint(contract)}",
            warn_only,
        )
        ok = False

    for group in contract.required_any_of:
        if not group:
            continue
        if not any(col in columns for col in group):
            _warn_or_raise(
                f"[{contract.name}] Expected at least one of columns: {group}.{_format_hint(contract)}",
                warn_only,
            )
            ok = False

    if contract.sample_model is not None and not df.empty:
        sample = df.iloc[0].to_dict()
        try:
            contract.sample_model.model_validate(sample)
        except ValidationError as exc:
            _warn_or_raise(
                f"[{contract.name}] Sample row validation warning: {exc.errors()}.{_format_hint(contract)}",
                warn_only,
            )
            ok = False

    return ok
