# Validation and source provenance

## Recorded AWS results

The evidence images and workbook describe the completed AWS/CloudShell run on October 24, 2026:

- Full classroom rubric test score: **120/120**.
- X-Ray map containing NovaMart-Orchestrator, workers, and Knowledge Base nodes.
- Hosted arithmetic example returning **$134.96**.

The screenshots are original evidence, not generated images. Account and resource identifiers may be visible; they are not AWS authentication credentials.

## GitHub source provenance

This repository was assembled from the completed implementation available in the shared workspace. The final deployed source remained in the user's CloudShell. The CommunicationAgent rounding instruction was reapplied to the workspace source while preparing this repository. A byte-for-byte match to that CloudShell file has not been established.

The supplied support modules, infrastructure, and classroom tests were retained. Account-specific `.env`, CLI targets, generated deployment state, dependency directories, and old phase-status reports were excluded.

The historical 120/120 score does not certify this exact commit. Local checks performed during publication are recorded below; live AWS tests must be repeated in an authenticated environment with the project's resources.

## Remaining limitations

A provisioned Memory resource is not a complete conversation memory integration. The aggregate service map does not prove span nesting. Refund state changes are not payment settlement. Prompt-only monetary rounding is not deterministic financial arithmetic. The educational identity supplied to a request is not production authentication.

## Publication checks

The repository copy passed all **25 offline tests** across `test_task2_offline.py`, `test_phase2_offline.py`, and `test_phase4_offline.py` during publication. These use mocks and DynamoDB emulation; they do not invoke live Bedrock models or certify deployed AWS state.
