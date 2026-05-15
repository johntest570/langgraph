"""Test spec modules for each checkpointer capability."""

import functools
import os

from langgraph.checkpoint.conformance.spec.test_copy_thread import (
    run_copy_thread_tests,
)
from langgraph.checkpoint.conformance.spec.test_delete_for_runs import (
    run_delete_for_runs_tests as _run_delete_for_runs_tests,
)
from langgraph.checkpoint.conformance.spec.test_delete_thread import (
    run_delete_thread_tests as _run_delete_thread_tests,
)
from langgraph.checkpoint.conformance.spec.test_get_tuple import run_get_tuple_tests
from langgraph.checkpoint.conformance.spec.test_list import run_list_tests
from langgraph.checkpoint.conformance.spec.test_prune import run_prune_tests
from langgraph.checkpoint.conformance.spec.test_put import run_put_tests
from langgraph.checkpoint.conformance.spec.test_put_writes import run_put_writes_tests


class HITLApprovalDeniedError(RuntimeError):
    """Raised when a human operator denies approval for a risky operation."""


def _hitl_approval_required(operation_name: str, fn):
    """Wrap *fn* so that a human operator must explicitly approve before it runs.

    Approval can be granted in two ways (in order of precedence):

    1. Set the environment variable ``HITL_APPROVED_OPS`` to a comma-separated
       list of operation names that are pre-approved (useful for CI pipelines
       where a human has already reviewed and signed off).
    2. Run interactively: the wrapper will prompt the operator via stdin.

    If approval is denied the wrapped function is **not** called and a
    :class:`HITLApprovalDeniedError` is raised instead.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # Check for pre-approved operations supplied via environment variable.
        pre_approved = {
            op.strip().lower()
            for op in os.environ.get("HITL_APPROVED_OPS", "").split(",")
            if op.strip()
        }
        if operation_name.lower() in pre_approved:
            return fn(*args, **kwargs)

        # Fall back to interactive prompt when a TTY is available.
        if os.isatty(0):  # stdin is a terminal
            answer = input(
                f"[HITL] The risky operation '{operation_name}' requires human "
                "approval.\n"
                "Type 'yes' to approve or anything else to deny: "
            ).strip().lower()
            if answer == "yes":
                return fn(*args, **kwargs)
            raise HITLApprovalDeniedError(
                f"Human operator denied approval for operation '{operation_name}'."
            )

        # Non-interactive environment without pre-approval — deny by default.
        raise HITLApprovalDeniedError(
            f"Operation '{operation_name}' requires HITL approval. "
            "Set the HITL_APPROVED_OPS environment variable to include "
            f"'{operation_name}' to grant pre-approval."
        )

    return wrapper


# Delete operations are wrapped with HITL approval to prevent accidental data loss.
run_delete_for_runs_tests = _hitl_approval_required(
    "run_delete_for_runs_tests", _run_delete_for_runs_tests
)
run_delete_thread_tests = _hitl_approval_required(
    "run_delete_thread_tests", _run_delete_thread_tests
)

__all__ = [
    "run_put_tests",
    "run_put_writes_tests",
    "run_get_tuple_tests",
    "run_list_tests",
    "run_delete_thread_tests",
    "run_delete_for_runs_tests",
    "run_copy_thread_tests",
    "run_prune_tests",
    "HITLApprovalDeniedError",
]
