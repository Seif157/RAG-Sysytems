"""Executable enforcement of the Dependency Rule (architecture spec section 4).

``import-linter`` enforces the same contracts in CI. This suite exists alongside
it for two reasons: it runs inside the ordinary ``pytest`` invocation, so a
violation surfaces in the same place as every other failure, and it reports the
offending file and line rather than a module pair.

Layering that is only documented is layering that erodes. These tests are the
difference between an architecture and a diagram.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

pytestmark = pytest.mark.architecture

SRC = Path(__file__).resolve().parents[2] / "src"
PACKAGE_ROOT = SRC / "rag"

#: Third-party distributions the domain is permitted to import: none.
_STDLIB = set(sys.stdlib_module_names)


@dataclass(frozen=True)
class Import:
    """One import statement found in the source tree."""

    module: str
    file: Path
    line: int

    @property
    def top_level(self) -> str:
        """The distribution or top-level package being imported."""
        return self.module.split(".", 1)[0]

    def __str__(self) -> str:
        """Render as a reviewer-friendly location."""
        return f"{self.file.relative_to(SRC)}:{self.line} imports {self.module!r}"


def _imports_of(layer: str) -> list[Import]:
    """Collect every module imported anywhere under ``rag.<layer>``."""
    found: list[Import] = []
    for path in sorted((PACKAGE_ROOT / layer).rglob("*.py")):
        # utf-8-sig so a stray byte-order mark cannot break the guard itself.
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.extend(Import(alias.name, path, node.lineno) for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.append(Import(node.module, path, node.lineno))
    return found


def _violations(layer: str, forbidden: set[str]) -> list[str]:
    """Return readable descriptions of every forbidden import in a layer."""
    return [
        str(imported)
        for imported in _imports_of(layer)
        if imported.module.split(".")[0] == "rag"
        and any(
            imported.module == banned or imported.module.startswith(f"{banned}.")
            for banned in forbidden
        )
    ]


class TestDomainPurity:
    """The domain is the centre: it depends on nothing."""

    def test_domain_imports_no_third_party_package(self):
        # The moment the domain imports pydantic, Qdrant or LlamaIndex, the core
        # is coupled to a release cycle it does not control. The prototype broke
        # exactly this way when LlamaIndex removed ServiceContext (ADR-002).
        offenders = [
            str(imported)
            for imported in _imports_of("domain")
            if imported.top_level not in _STDLIB and imported.top_level != "rag"
        ]

        assert offenders == []

    def test_domain_imports_no_other_layer(self):
        assert (
            _violations(
                "domain",
                {
                    "rag.application",
                    "rag.presentation",
                    "rag.infrastructure",
                    "rag.core",
                    "rag.config",
                    "rag.workers",
                    "rag.evaluation",
                },
            )
            == []
        )

    def test_the_domain_package_imports_cleanly_on_its_own(self):
        # A guard against a cycle or a hidden dependency that only the import
        # machinery would notice.
        import importlib

        assert importlib.import_module("rag.domain.models") is not None
        assert importlib.import_module("rag.domain.ports") is not None


class TestApplicationIsolation:
    def test_application_does_not_depend_on_infrastructure(self):
        # Use cases must be executable against fakes with zero network.
        assert (
            _violations(
                "application",
                {"rag.infrastructure", "rag.presentation", "rag.workers", "rag.config"},
            )
            == []
        )

    def test_application_imports_no_vendor_sdk(self):
        banned = {
            "qdrant_client",
            "llama_index",
            "openai",
            "google",
            "anthropic",
            "celery",
            "redis",
            "sqlalchemy",
            "fastapi",
            "streamlit",
        }
        offenders = [
            str(imported) for imported in _imports_of("application") if imported.top_level in banned
        ]

        assert offenders == []


class TestInfrastructureIsolation:
    def test_infrastructure_does_not_reach_back_into_orchestration(self):
        assert _violations("infrastructure", {"rag.application", "rag.presentation"}) == []


class TestPresentationIsolation:
    def test_presentation_does_not_import_infrastructure_directly(self):
        # Routers resolve use cases from the composition root; the UI must not
        # know that Qdrant exists.
        assert _violations("presentation", {"rag.infrastructure"}) == []


class TestConfigIsALeaf:
    def test_config_imports_no_other_layer(self):
        assert (
            _violations(
                "config",
                {
                    "rag.domain",
                    "rag.application",
                    "rag.presentation",
                    "rag.infrastructure",
                    "rag.core",
                    "rag.workers",
                },
            )
            == []
        )


class TestUtilsIsALeaf:
    def test_utils_knows_nothing_about_the_system(self):
        assert (
            _violations(
                "utils",
                {
                    "rag.domain",
                    "rag.application",
                    "rag.presentation",
                    "rag.infrastructure",
                    "rag.core",
                    "rag.config",
                    "rag.workers",
                },
            )
            == []
        )


class TestEntryPointsAreNotImported:
    @pytest.mark.parametrize(
        "layer",
        ["domain", "application", "infrastructure", "core", "config"],
    )
    def test_nothing_imports_the_streamlit_entry_point(self, layer):
        # The UI is an entry point. Nothing imports an entry point, and doing so
        # would execute the whole Streamlit script as a side effect.
        assert _violations(layer, {"rag.presentation"}) == []


class TestGuardsAreRealisticallyWired:
    def test_the_scanner_finds_the_imports_that_do_exist(self):
        # A guard that silently scans nothing passes forever. This proves the
        # AST walk is actually reading source.
        assert any(imported.module.startswith("rag.domain") for imported in _imports_of("core"))

    def test_the_scanner_would_catch_a_violation(self):
        # Proves _violations matches on prefix, not equality alone.
        assert _violations("core", {"rag.domain"}) != []
