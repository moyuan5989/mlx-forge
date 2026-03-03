"""Curated dataset catalog for LMForge.

Contains ~20 known-good datasets across categories:
general, conversation, code, math, reasoning, safety, domain.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ColumnMapping:
    """Defines how to convert HF dataset columns to LMForge standard format."""
    type: str  # "rename", "chat_messages", "alpaca", "sharegpt", "preference", "text_column"
    mapping: dict[str, str] = field(default_factory=dict)


@dataclass
class DatasetProfile:
    """A curated dataset entry in the catalog."""
    id: str                    # LMForge internal ID
    source: str                # HF Hub dataset ID
    display_name: str
    category: str              # general, code, math, conversation, reasoning, safety, domain
    format: str                # chat, completions, text, preference
    description: str
    license: str
    total_samples: int
    avg_tokens: int
    tags: list[str] = field(default_factory=list)
    split: str = "train"
    subset: str | None = None  # HF dataset config/subset name
    columns: ColumnMapping | None = None  # How to convert to LMForge format

    def to_dict(self) -> dict:
        """Convert to dict for API responses."""
        d = {
            "id": self.id,
            "source": self.source,
            "display_name": self.display_name,
            "category": self.category,
            "format": self.format,
            "description": self.description,
            "license": self.license,
            "total_samples": self.total_samples,
            "avg_tokens": self.avg_tokens,
            "tags": self.tags,
            "split": self.split,
            "subset": self.subset,
        }
        if self.columns:
            d["columns"] = {"type": self.columns.type, "mapping": self.columns.mapping}
        return d


# ── Curated Catalog ──────────────────────────────────────────────────────────

DATASET_CATALOG: dict[str, DatasetProfile] = {

    # ── General Instruction ──

    "alpaca-cleaned": DatasetProfile(
        id="alpaca-cleaned",
        source="yahma/alpaca-cleaned",
        display_name="Alpaca (Cleaned)",
        category="general",
        format="completions",
        total_samples=51760,
        avg_tokens=85,
        license="CC-BY-4.0",
        tags=["english", "instruction"],
        description="Cleaned Stanford Alpaca. Solid general-purpose base.",
        columns=ColumnMapping(
            type="alpaca",
            mapping={"instruction": "instruction", "input": "input", "output": "output"},
        ),
    ),

    "dolly-15k": DatasetProfile(
        id="dolly-15k",
        source="databricks/databricks-dolly-15k",
        display_name="Dolly 15K",
        category="general",
        format="completions",
        total_samples=15011,
        avg_tokens=110,
        license="CC-BY-SA-3.0",
        tags=["english", "instruction"],
        description="Databricks employee-generated instruction dataset.",
        columns=ColumnMapping(
            type="rename",
            mapping={"instruction": "prompt", "response": "completion"},
        ),
    ),

    "openhermes-25": DatasetProfile(
        id="openhermes-25",
        source="teknium/OpenHermes-2.5",
        display_name="OpenHermes 2.5",
        category="general",
        format="chat",
        total_samples=1000000,
        avg_tokens=280,
        license="Apache-2.0",
        tags=["english", "instruction", "multi-turn"],
        description="1M high-quality synthetic conversations. Sample for Apple Silicon.",
        columns=ColumnMapping(
            type="sharegpt",
            mapping={"conversations": "conversations"},
        ),
    ),

    "slimorca-dedup": DatasetProfile(
        id="slimorca-dedup",
        source="Open-Orca/SlimOrca-Dedup",
        display_name="SlimOrca (Dedup)",
        category="general",
        format="chat",
        total_samples=362000,
        avg_tokens=200,
        license="MIT",
        tags=["english", "instruction"],
        description="Deduplicated SlimOrca. Clean general-purpose conversations.",
        columns=ColumnMapping(
            type="chat_messages",
            mapping={"conversations": "messages"},
        ),
    ),

    # ── Conversation / Chat ──

    "oasst-guanaco": DatasetProfile(
        id="oasst-guanaco",
        source="timdettmers/openassistant-guanaco",
        display_name="OpenAssistant Guanaco",
        category="conversation",
        format="text",
        total_samples=9846,
        avg_tokens=256,
        license="Apache-2.0",
        tags=["english", "multi-turn"],
        description="Filtered OASST1 subset. High quality multi-turn dialogue.",
        columns=ColumnMapping(
            type="text_column",
            mapping={"text": "text"},
        ),
    ),

    "ultrachat-200k": DatasetProfile(
        id="ultrachat-200k",
        source="HuggingFaceH4/ultrachat_200k",
        display_name="UltraChat 200K",
        category="conversation",
        format="chat",
        total_samples=200000,
        avg_tokens=350,
        license="MIT",
        tags=["english", "multi-turn", "synthetic"],
        description="200K synthetic multi-turn conversations across diverse topics.",
        columns=ColumnMapping(
            type="chat_messages",
            mapping={"messages": "messages"},
        ),
        split="train_sft",
    ),

    "capybara-9k": DatasetProfile(
        id="capybara-9k",
        source="LDJnr/Capybara",
        display_name="Capybara 9K",
        category="conversation",
        format="chat",
        total_samples=16000,
        avg_tokens=400,
        license="Apache-2.0",
        tags=["english", "multi-turn"],
        description="Multi-turn conversations from diverse sources.",
        columns=ColumnMapping(type="sharegpt"),
    ),

    # ── Code ──

    "magicoder-oss": DatasetProfile(
        id="magicoder-oss",
        source="ise-uiuc/Magicoder-OSS-Instruct-75K",
        display_name="Magicoder OSS-Instruct",
        category="code",
        format="completions",
        total_samples=75000,
        avg_tokens=340,
        license="MIT",
        tags=["code", "python", "instruction"],
        description="Synthetic code instruction data from OSS snippets.",
        columns=ColumnMapping(
            type="rename",
            mapping={"problem": "prompt", "solution": "completion"},
        ),
    ),

    "code-alpaca-20k": DatasetProfile(
        id="code-alpaca-20k",
        source="sahil2801/CodeAlpaca-20k",
        display_name="Code Alpaca 20K",
        category="code",
        format="completions",
        total_samples=20022,
        avg_tokens=150,
        license="Apache-2.0",
        tags=["code", "instruction"],
        description="Code instruction-following based on Alpaca format.",
        columns=ColumnMapping(
            type="alpaca",
            mapping={"instruction": "instruction", "input": "input", "output": "output"},
        ),
    ),

    "glaive-code-assistant": DatasetProfile(
        id="glaive-code-assistant",
        source="glaiveai/glaive-code-assistant-v2",
        display_name="Glaive Code Assistant",
        category="code",
        format="chat",
        total_samples=136000,
        avg_tokens=450,
        license="Apache-2.0",
        tags=["code", "multi-turn", "instruction"],
        description="Multi-turn code conversations with system prompts.",
        columns=ColumnMapping(
            type="chat_messages",
            mapping={"conversations": "messages"},
        ),
    ),

    # ── Math / Reasoning ──

    "metamathqa": DatasetProfile(
        id="metamathqa",
        source="meta-math/MetaMathQA",
        display_name="MetaMathQA",
        category="math",
        format="completions",
        total_samples=395000,
        avg_tokens=200,
        license="MIT",
        tags=["math", "reasoning"],
        description="Augmented math reasoning dataset. Strong math performance.",
        columns=ColumnMapping(
            type="rename",
            mapping={"query": "prompt", "response": "completion"},
        ),
    ),

    "orca-math": DatasetProfile(
        id="orca-math",
        source="microsoft/orca-math-word-problems-200k",
        display_name="Orca Math 200K",
        category="math",
        format="completions",
        total_samples=200000,
        avg_tokens=180,
        license="MIT",
        tags=["math", "word-problems"],
        description="Math word problems with step-by-step solutions.",
        columns=ColumnMapping(
            type="rename",
            mapping={"question": "prompt", "answer": "completion"},
        ),
    ),

    # ── Reasoning ──

    "arc-challenge": DatasetProfile(
        id="arc-challenge",
        source="allenai/ai2_arc",
        display_name="ARC Challenge",
        category="reasoning",
        format="completions",
        total_samples=2590,
        avg_tokens=80,
        license="CC-BY-SA-4.0",
        tags=["reasoning", "science", "qa"],
        subset="ARC-Challenge",
        description="Science reasoning questions. Good eval/calibration set.",
        columns=ColumnMapping(
            type="rename",
            mapping={"question": "prompt", "answerKey": "completion"},
        ),
    ),

    # ── Safety / Alignment ──

    "safe-rlhf": DatasetProfile(
        id="safe-rlhf",
        source="PKU-Alignment/PKU-SafeRLHF",
        display_name="PKU SafeRLHF",
        category="safety",
        format="preference",
        total_samples=44000,
        avg_tokens=200,
        license="CC-BY-NC-4.0",
        tags=["safety", "preference", "dpo"],
        description="Safety preference data for RLHF/DPO alignment.",
        columns=ColumnMapping(type="preference"),
    ),

    "hh-rlhf": DatasetProfile(
        id="hh-rlhf",
        source="Anthropic/hh-rlhf",
        display_name="Anthropic HH-RLHF",
        category="safety",
        format="preference",
        total_samples=161000,
        avg_tokens=250,
        license="MIT",
        tags=["safety", "preference", "dpo", "helpfulness"],
        description="Helpfulness and harmlessness preference data from Anthropic.",
        columns=ColumnMapping(type="preference"),
    ),

    "ultrafeedback": DatasetProfile(
        id="ultrafeedback",
        source="HuggingFaceH4/ultrafeedback_binarized",
        display_name="UltraFeedback (Binarized)",
        category="safety",
        format="preference",
        total_samples=60000,
        avg_tokens=300,
        license="MIT",
        tags=["preference", "dpo", "quality"],
        description="Binarized preference data from UltraFeedback. Used to train Zephyr.",
        columns=ColumnMapping(type="preference"),
    ),

    # ── Domain-Specific ──

    "medical-meadow": DatasetProfile(
        id="medical-meadow",
        source="medalpaca/medical_meadow_medqa_4options",
        display_name="Medical Meadow (MedQA)",
        category="domain",
        format="completions",
        total_samples=10178,
        avg_tokens=120,
        license="GPL-3.0",
        tags=["medical", "qa"],
        description="Medical QA from USMLE-style questions.",
        columns=ColumnMapping(
            type="rename",
            mapping={"input": "prompt", "output": "completion"},
        ),
    ),

    "finance-alpaca": DatasetProfile(
        id="finance-alpaca",
        source="gbharti/finance-alpaca",
        display_name="Finance Alpaca",
        category="domain",
        format="completions",
        total_samples=68000,
        avg_tokens=130,
        license="CC-BY-4.0",
        tags=["finance", "instruction"],
        description="Finance domain instruction-following data.",
        columns=ColumnMapping(
            type="alpaca",
            mapping={"instruction": "instruction", "input": "input", "output": "output"},
        ),
    ),
}
