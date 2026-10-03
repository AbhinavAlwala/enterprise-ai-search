import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from enterprise_ai_search.hybrid import HybridIndex, HybridResult
from enterprise_ai_search.reranker import CANDIDATE_DEPTH

MAX_GROUPS = 32
MAX_GROUP_HEADER_LENGTH = 2048


def validate_identifier(value: object, maximum: int = 64) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", value):
        raise ValueError("Invalid identifier")
    if len(value) > maximum:
        raise ValueError("Invalid identifier")
    return value


def validate_members(values: frozenset[str]) -> None:
    if not isinstance(values, frozenset):
        raise ValueError("Membership must be a frozen set")
    for value in values:
        validate_identifier(value)


@dataclass(frozen=True)
class PrincipalContext:
    tenant_id: str
    principal_id: str
    groups: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        validate_identifier(self.tenant_id)
        validate_identifier(self.principal_id)
        validate_members(self.groups)
        if len(self.groups) > MAX_GROUPS:
            raise ValueError("Too many groups")


def parse_principal(tenant: str | None, principal: str | None, groups: str | None) -> PrincipalContext:
    if tenant is None or principal is None:
        raise ValueError("Missing identity")
    memberships: frozenset[str] = frozenset()
    if groups is not None and len(groups) > MAX_GROUP_HEADER_LENGTH:
        raise ValueError("Group header too long")
    if groups is not None and groups.strip():
        items = groups.split(",")
        if len(items) > MAX_GROUPS:
            raise ValueError("Too many groups")
        memberships = frozenset(item.strip() for item in items)
    return PrincipalContext(tenant.strip(), principal.strip(), memberships)


@dataclass(frozen=True)
class DocumentAccessPolicy:
    document_id: str
    tenant_id: str
    visibility: Literal["tenant", "restricted"]
    allowed_principals: frozenset[str] = frozenset()
    allowed_groups: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        validate_identifier(self.document_id, maximum=128)
        validate_identifier(self.tenant_id)
        validate_members(self.allowed_principals)
        validate_members(self.allowed_groups)
        if self.visibility not in ("tenant", "restricted"):
            raise ValueError("Invalid visibility")
        if self.visibility == "tenant" and (self.allowed_principals or self.allowed_groups):
            raise ValueError("Tenant visibility cannot also specify allowlists")


@dataclass(frozen=True)
class PolicyStore:
    tenants: frozenset[str]
    policies: Mapping[str, DocumentAccessPolicy] = field(repr=False)

    def __post_init__(self) -> None:
        validate_members(self.tenants)
        if not self.tenants or not isinstance(self.policies, Mapping):
            raise ValueError("Invalid policy store")
        for document_id, policy in self.policies.items():
            if not isinstance(policy, DocumentAccessPolicy):
                raise ValueError("Invalid document policy")
            if document_id != policy.document_id or policy.tenant_id not in self.tenants:
                raise ValueError("Inconsistent document policy")
        object.__setattr__(self, "policies", MappingProxyType(dict(self.policies)))

    def allows(self, document_id: str, principal: PrincipalContext) -> bool:
        policy = self.policies.get(document_id)
        if principal.tenant_id not in self.tenants or policy is None or policy.tenant_id != principal.tenant_id:
            return False
        return (policy.visibility == "tenant" or principal.principal_id in policy.allowed_principals
                or bool(principal.groups & policy.allowed_groups))


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate policy field")
        result[key] = value
    return result


def _members(value: object) -> frozenset[str]:
    if not isinstance(value, list):
        raise ValueError("Allowlist must be an array")
    members = frozenset(validate_identifier(item) for item in value)
    if len(members) != len(value):
        raise ValueError("Duplicate membership")
    return members


def load_policy_store(path: Path) -> PolicyStore:
    data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    if not isinstance(data, dict) or set(data) != {"schema_version", "description", "tenants", "documents"}:
        raise ValueError("Invalid policy file fields")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1 or not isinstance(data["description"], str):
        raise ValueError("Invalid policy file schema")
    tenants = _members(data["tenants"])
    if not isinstance(data["documents"], list):
        raise ValueError("Documents must be an array")
    policies = {}
    for record in data["documents"]:
        if not isinstance(record, dict) or set(record) != {
            "document_id", "tenant_id", "visibility", "allowed_principals", "allowed_groups",
        }:
            raise ValueError("Invalid document policy fields")
        policy = DocumentAccessPolicy(
            record["document_id"], record["tenant_id"], record["visibility"],
            _members(record["allowed_principals"]), _members(record["allowed_groups"]),
        )
        if policy.document_id in policies:
            raise ValueError("Duplicate document policy")
        policies[policy.document_id] = policy
    return PolicyStore(tenants, policies)


@dataclass(frozen=True)
class AuthorizedIndex:
    index: HybridIndex = field(repr=False)
    store: PolicyStore = field(repr=False)
    principal: PrincipalContext = field(repr=False)

    def search(self, query: str, top_k: int = 5) -> list[HybridResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        candidates = self.index.search(query, CANDIDATE_DEPTH)
        authorized = []
        for candidate in candidates[:CANDIDATE_DEPTH]:
            if not self.store.allows(candidate.document_id, self.principal):
                continue
            representatives = [item for item in (candidate.bm25, candidate.dense) if item is not None]
            if not representatives or any(
                item.document_id != candidate.document_id or item.passage.document_id != candidate.document_id
                for item in representatives
            ):
                continue
            # Frozen reranking requires consecutive ranks after denied candidates are removed.
            authorized.append(replace(candidate, rank=len(authorized) + 1))
        return authorized[:top_k]
