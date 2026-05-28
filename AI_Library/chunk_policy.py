import os
from typing import Dict

from langchain.text_splitter import RecursiveCharacterTextSplitter


CHUNK_POLICIES: Dict[str, Dict[str, int]] = {
    "baseline_800_100": {"chunk_size": 800, "chunk_overlap": 100},
    "small_500_100": {"chunk_size": 500, "chunk_overlap": 100},
}

DEFAULT_POLICY = "small_500_100"


def resolve_chunk_policy(policy_name: str | None = None) -> Dict[str, int]:
    configured = (policy_name or os.getenv("AI_CHUNK_POLICY", DEFAULT_POLICY)).strip().lower()
    if configured not in CHUNK_POLICIES:
        configured = DEFAULT_POLICY
    policy = CHUNK_POLICIES[configured]
    return {
        "name": configured,
        "chunk_size": policy["chunk_size"],
        "chunk_overlap": policy["chunk_overlap"],
    }


def build_text_splitter(policy_name: str | None = None) -> RecursiveCharacterTextSplitter:
    policy = resolve_chunk_policy(policy_name)
    return RecursiveCharacterTextSplitter(
        chunk_size=policy["chunk_size"],
        chunk_overlap=policy["chunk_overlap"],
    )
