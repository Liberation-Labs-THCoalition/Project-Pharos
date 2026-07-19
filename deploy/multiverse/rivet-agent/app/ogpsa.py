"""OGPSA — Orthogonal Gradient Persona Survival Architecture.

Protects the agent's persona from adversarial prompt injection by
detecting and filtering identity-modifying content. Based on the
OGPSA/SLERP research: gradient survival gate projects away from
persona-destabilizing directions.

Reference: Liberation-Labs-THCoalition OGPSA research
"""

import re
import logging

log = logging.getLogger("ogpsa")

# Patterns that signal persona manipulation attempts
ADVERSARIAL_PATTERNS = [
    r"ignore\s+(previous|all|your)\s+(instructions|persona|identity)",
    r"you\s+are\s+now\s+(?:a|an)\s+",
    r"forget\s+(?:everything|your|all)",
    r"new\s+(?:identity|persona|role|character)",
    r"pretend\s+(?:to\s+be|you\s+are)",
    r"act\s+as\s+(?:if|though)\s+you",
    r"override\s+(?:your|system|persona)",
    r"disregard\s+(?:your|the)\s+(?:system|persona)",
    r"jailbreak",
    r"DAN\s+mode",
]

COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in ADVERSARIAL_PATTERNS]


class OGPSAGate:
    """Filters adversarial inputs that target persona identity."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.blocked_count = 0

    def filter(self, message: str) -> str:
        """Check message for persona-adversarial content.

        Returns the message unchanged if clean, or strips adversarial
        segments. Does NOT block the entire message — only the attack vector.
        """
        if not self.enabled:
            return message

        for pattern in COMPILED_PATTERNS:
            if pattern.search(message):
                log.warning(f"OGPSA: adversarial pattern detected, filtering")
                self.blocked_count += 1
                message = pattern.sub("[filtered]", message)

        return message

    def is_adversarial(self, message: str) -> bool:
        """Check without filtering — returns True if any pattern matches."""
        return any(p.search(message) for p in COMPILED_PATTERNS)
