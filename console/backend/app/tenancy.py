"""Partitioning for work submitted under a shared service identity.

A normal Console account is its own blast radius: the training circuit
breaker and the pending-job cap are keyed on ``users.id``, so one account's
failures can only ever hurt that account.

A **privileged service-key caller is different**. One service credential
authenticates one synthetic account (``service_user_email``), and that single
account submits training on behalf of the caller's ENTIRE user base. Keying a
per-user control on the account alone therefore turns it into a control over
all of that caller's users at once:

* ``FAILURE_THRESHOLD`` consecutive failures from ONE of its users pause the
  queue for EVERY one of them, and the pause is written with
  ``next_attempt_at = NULL`` so nothing releases it automatically; and
* ``PER_USER_MAX_PENDING`` counts all of their jobs together, so one user is
  refused because of a stranger's queue.

So a service-key caller presents an opaque partition key -- a header value it
derives one-way from its own user id -- and the queue keys those controls on
``(user_id, tenant_key)`` instead of ``user_id`` alone. The key is opaque on
purpose: it is a partition, not an identity, and it lets the Console keep a
per-user control per-user without ever learning who the upstream user is.

The requirement is deliberately one-directional. An ordinary account keeps
``ACCOUNT_TENANT`` (the empty key) and is unaffected. A service-key request
that names no tenant is REFUSED rather than folded into a shared partition:
work whose owner is unknown must fail loudly, because the silent version of
that fallback is the exact defect this module exists to remove.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Wire name of the partition key on the privileged service-key path. This is
# the contract with the upstream integration; changing it re-partitions live
# queues, so it is a constant rather than a literal at each read site.
SERVICE_TENANT_HEADER = "X-Viola-Tenant"

# The partition an ordinary account occupies: itself. Empty rather than NULL
# so it participates in a composite PRIMARY KEY without NULL comparison
# semantics (in SQLite every NULL is distinct, which would defeat the key).
ACCOUNT_TENANT = ""

# Opaque, bounded, and case-fixed. Bounded so a caller cannot mint unbounded
# partitions or smuggle a payload through a header the queue stores verbatim;
# case-fixed so two spellings of one key cannot become two partitions and
# silently split a user's own breaker.
TENANT_KEY_PATTERN = re.compile(r"\A[a-z0-9]{16,64}\Z")


class MissingServiceTenantError(ValueError):
    """A service-key request named no tenant, or named an unusable one.

    Raised instead of returning ``ACCOUNT_TENANT``: falling back to the shared
    partition is what makes one user's failures everyone's failures.
    """


def normalize_service_tenant_key(raw: str | None) -> str:
    """Return the validated partition key carried by a service-key request.

    Raises :class:`MissingServiceTenantError` when the header is absent, blank
    or malformed. There is no permissive branch by design.
    """
    candidate = (raw or "").strip().lower()
    if not candidate:
        raise MissingServiceTenantError(
            f"{SERVICE_TENANT_HEADER} is required on the service-key path: a training "
            "job submitted under the shared service account with no tenant would put "
            "every upstream user behind one circuit breaker and one pending-job cap."
        )
    if not TENANT_KEY_PATTERN.match(candidate):
        raise MissingServiceTenantError(
            f"{SERVICE_TENANT_HEADER} must be 16-64 lowercase alphanumeric characters "
            "(an opaque partition key, not an account identifier)."
        )
    return candidate


@dataclass(frozen=True, slots=True)
class QueuePartition:
    """The scope a queue control applies to.

    ``user_id`` alone is the scope for an ordinary account. For work arriving
    under a shared service identity the scope is the pair, so the breaker,
    the pending-job cap and the failure backoff each apply to exactly one
    upstream user.

    Frozen so it can key the in-memory retry-timer map: those timers were
    keyed on ``user_id`` too, which is the same globality bug wearing a
    different hat (one tenant's backoff suppressed every other tenant's
    refill).
    """

    user_id: int
    tenant_key: str = ACCOUNT_TENANT

    @classmethod
    def for_account(cls, user_id: int) -> QueuePartition:
        """The partition of an ordinary account: the account itself."""
        return cls(user_id=int(user_id), tenant_key=ACCOUNT_TENANT)

    @classmethod
    def coerce(cls, value: QueuePartition | int) -> QueuePartition:
        """Accept a partition, or a bare account id meaning ``for_account``.

        Only read-and-clear paths (breaker state, listing, resume) take the
        bare form, where an account id is unambiguous. Job submission does
        not: it takes a partition and nothing else, so no caller can enqueue
        work whose partition was inferred.
        """
        if isinstance(value, QueuePartition):
            return value
        return cls.for_account(value)

    @property
    def is_service_tenant(self) -> bool:
        """True when this partition names an upstream tenant, not an account."""
        return self.tenant_key != ACCOUNT_TENANT

    def __str__(self) -> str:  # pragma: no cover - logging convenience
        if not self.is_service_tenant:
            return f"user {self.user_id}"
        return f"user {self.user_id} tenant {self.tenant_key}"


__all__ = [
    "ACCOUNT_TENANT",
    "SERVICE_TENANT_HEADER",
    "TENANT_KEY_PATTERN",
    "MissingServiceTenantError",
    "QueuePartition",
    "normalize_service_tenant_key",
]
