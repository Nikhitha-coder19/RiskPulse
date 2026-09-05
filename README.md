# RiskPulse

## 📌 Availability Note

I understand that the Razorpay AI Builder Internship is intended to be a full-time, in-person opportunity in Bangalore starting in September 2026.

Due to my university's academic requirements, I am unable to begin a full-time internship before late November/early December 2026. I would therefore be sincerely grateful if my project could still be evaluated based on the work demonstrated in this repository rather than being excluded solely because of my current joining-date constraint.

If I am shortlisted, I will make every reasonable effort to seek an exception from my university for an earlier joining date. If an earlier release is not possible, I can commit to joining from late November/early December 2026, when my university permits full-time internships.

I completely understand if the September start is a firm requirement. I simply wanted to communicate my situation transparently and respectfully request that the project be considered on its technical merit before a decision is made based on availability.

## Overview

RiskPulse is a multi-layer AI risk decisioning and operations system designed to help merchants reduce losses from fraudulent transactions, coordinated behavioral abuse, and merchant-level risk.

It is designed from the perspective of internal Razorpay operations and risk analysts. The prototype provides an authenticated operations console, not a customer-facing merchant portal.

RiskPulse combines three independent risk signals:

1. Traditional transaction-level fraud risk
2. Behavioral and coordinated fraud risk
3. Merchant-level risk

The three signals are fused into one final operational risk score and action.

## Problem

Merchant loss prevention requires more than identifying isolated suspicious transactions. RiskPulse addresses:

- Individual fraudulent transactions
- Coordinated or behavioral abuse patterns
- Risky merchant behavior
- Transactions requiring manual analyst review
- Customer verification and challenge flows
- Operational monitoring
- Retrospective analyst feedback

## Solution

The active application is an internal Streamlit operations console. It lets authenticated employees inspect transactions, investigate risk decisions, manage review cases, monitor challenge lifecycles, and record retrospective feedback.

Customer verification is external to this prototype. Operations employees monitor the resulting lifecycle events in RiskPulse; they do not perform customer verification inside the application.

## Architecture

```text
Transaction
    |
    v
Feature preparation and model adapters
    |
    +--> Traditional transaction model
    |
    +--> Behavioral/coordinated risk model
    |
    +--> Merchant-level risk model
             |
             v
        Fusion engine
             |
             v
      Final risk score
             |
             v
      Operational action
             |
             v
       Runtime state update
```

The runtime lifecycle is:

```text
READ STATE
    |
    v
CALCULATE BEFORE FEATURES
    |
    v
MODEL SCORES
    |
    v
FUSION
    |
    v
ACTION
    |
    v
UPDATE STATE
```

Runtime state supports the operational workflows and preserves the history needed for stateful behavioral and merchant features.

## Risk Decisioning

The fusion engine calculates:

```text
Final Risk Score =
    0.40 * P_traditional
  + 0.35 * P_behavioral
  + 0.25 * P_merchant
```

Final fused-score action boundaries:

| Final risk score | Action |
|---|---|
| `< 0.30` | `ALLOW` |
| `0.30` to `< 0.60` | `REVIEW` |
| `0.60` to `< 0.85` | `CHALLENGE` |
| `>= 0.85` | `BLOCK` |

These are final operational action thresholds. They are distinct from the model classification thresholds used within the individual model layers:

| Model layer | Classification threshold |
|---|---:|
| Traditional | `0.11` |
| Behavioral | `0.17` |
| Merchant | `0.27` |

A model classification threshold must not be interpreted as a final fused action boundary.

## ML Models

The following are frozen prototype/evaluation artifacts. The metrics are evaluation results, not production guarantees.

| Layer | Model | Features | Threshold | Precision | Recall | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| Traditional | Random Forest | 21 | 0.11 | 27.07% | 48.26% | 34.69% | 79.25% |
| Behavioral | Balanced Random Forest, 300 trees | 26 | 0.17 | 39.48% | 77.90% | 52.40% | 97.94% |
| Merchant | Logistic Regression | 19 | 0.27 | 30.83% | 81.56% | 44.74% | 87.11% |

Precision and recall reflect different operating tradeoffs. These figures should be read as prototype evaluation results on the associated evaluation data, not as claims about production performance or monetary savings.

## Operational Workflows

### Dashboard

The dashboard provides risk and decision overviews, review metrics, open review cases, and resolved reviews.

### Transaction Explorer

Employees can browse transactions, inspect their risk decisions, and open an investigation.

### Transaction Investigation

An investigation brings together:

- The risk decision
- Model-level risk signals
- Transaction details
- Customer protection status
- Analyst review outcome where applicable
- The operational next action
- Challenge status
- An analyst feedback entry point

### Review Queue

Review cases follow this lifecycle:

```text
OPEN -> IN_PROGRESS -> RESOLVED
```

Analysts can assign open cases. Assigned cases are handled by the assigned analyst, and resolution requires an analyst decision. Authorization is enforced service-side. The queue supports both active and historical review cases.

The Review Queue analyst decision is separate from the original RiskPulse AI decision.

### Challenge Monitoring

A `CHALLENGE` decision automatically creates a challenge. Operations employees monitor the challenge lifecycle and its events chronologically. Customer verification occurs externally, and its external lifecycle events enter RiskPulse for operational monitoring.

```text
Transaction
    |
    v
RiskPulse scoring
    |
    v
CHALLENGE decision
    |
    v
Challenge automatically created
    |
    v
Customer verification occurs externally
    |
    v
External lifecycle events enter RiskPulse
    |
    v
Operations monitors the challenge
```

### Analyst Feedback

Analyst feedback is retrospective and independent. RiskPulse keeps these concepts distinct:

1. The RiskPulse AI decision
2. The Review Queue analyst decision
3. Retrospective analyst feedback

Feedback does not overwrite the original AI decision, review decision, transaction fraud label, or challenge history.

## Customer Protection

RiskPulse translates internal risk decisions into high-level protection states where appropriate. Merchant protection remains the primary objective of RiskPulse while these protection states also help prevent legitimate customers from proceeding through risky or potentially harmful transactions.

| Internal action | Protection state |
|---|---|
| `ALLOW` | `PROCEED` |
| `REVIEW` | `REVIEW` |
| `CHALLENGE` | `VERIFY` |
| `BLOCK` | `STOP` |

Customer/Merchant-facing protection states should not expose internal model names, fusion weights, thresholds, raw probabilities, or internal reasoning. Merchant-risk warning signals may also be surfaced at an appropriate high level.

## Auditability

Operational actions are recorded with authenticated employee identity. Relevant audited operations include transaction investigation and feedback-, review-, and challenge-related operational activity. Audit records support accountability for employee actions without exposing credentials or secrets.

## Security

RiskPulse includes prototype employee authentication appropriate for the demonstrated system:

- Employee login
- Session-based authenticated identity
- PBKDF2-HMAC-SHA256 password hashing
- Per-user salt
- 120,000 PBKDF2 iterations
- Configurable session-token secret through `RISKPULSE_PROTOTYPE_SESSION_SECRET`

This is prototype authentication, not a claim of production-grade security. No passwords, session secrets, tokens, or private credentials are published.

## Setup

The intended evaluator flow is:

```powershell
git clone https://github.com/Nikhitha-coder19/RiskPulse.git
cd RiskPulse
python -m pip install -r requirements.txt
python scripts/provision_models.py
python ui/init_operations_db.py
python scripts/seed_demo_data.py
streamlit run ui/app.py
```

`ui/app.py` initializes the operations schema when the application starts. Running `python ui/init_operations_db.py` explicitly creates the local SQLite schema before seeding.

The application uses the local database at `runtime/riskpulse_state.db`. This database is generated local state and is not part of the public repository.

### Demo Access

RiskPulse includes preconfigured buildathon-only employee accounts for evaluating the internal operations console.

| Employee            | Role       | Password            |
| ------------------- | ---------- | ------------------- |
| `analyst.alex`      | Analyst    | `riskpulse-alex-2026`  |
| `analyst.priya`     | Analyst    | `riskpulse-priya-2026`  |
| `supervisor.morgan` | Supervisor | `riskpulse-morgan-2026` |

These credentials are for this RiskPulse prototype only. They do not provide access to any external systems or real customer or merchant accounts.


## Model Provisioning

The model binaries are intentionally not stored in normal Git history because of their size. The repository contains:

- `model/model_manifest.json`
- `scripts/provision_models.py`

The provisioning script downloads the required model artifacts from the public GitHub Release v1.0.0 and verifies each artifact using its expected file size and SHA-256 checksum before installing it locally.
Run the default provisioning command:

```powershell
python scripts/provision_models.py
```

Optional release overrides:

```powershell
python scripts/provision_models.py --tag v1.0.0
python scripts/provision_models.py --repository OWNER/REPOSITORY --tag RELEASE_TAG
```

The same values can be supplied with environment variables:

```powershell
$env:RISKPULSE_RELEASE_REPOSITORY = "OWNER/REPOSITORY"
$env:RISKPULSE_RELEASE_TAG = "RELEASE_TAG"
python scripts/provision_models.py
```

Provisioning:

- Downloads only the five required model assets
- Verifies each file size
- Verifies each SHA-256 checksum
- Skips an existing file when it is already valid
- Uses a temporary download file before replacement
- Fails on download, size, or checksum errors
- Does not download datasets
- Does not download runtime database state

Release assets:

```text
RiskPulse-random_forest.pkl
RiskPulse-behavioral_random_forest.pkl
RiskPulse-behavioral_imputer.pkl
RiskPulse-merchant_logistic_regression.pkl
RiskPulse-merchant_imputer.pkl
```

The default release tag is `v1.0.0`.

## Demo

Run the deterministic demo seed with:

```powershell
python scripts/seed_demo_data.py
```

The seed is deterministic, idempotent, prefix-scoped, and intended for demonstration and QA. It does not represent production data.

Expected demo action distribution:

```text
ALLOW = 3
REVIEW = 1
CHALLENGE = 2
BLOCK = 0
```

To remove only the demo records created by the seed:

```powershell
python scripts/seed_demo_data.py cleanup
```

The prototype includes these demo employee usernames:

```text
analyst.alex
analyst.priya
supervisor.morgan
```

Passwords are intentionally not published in this repository. Demo credentials should be supplied separately when needed.

## Repository Structure

```text
ui/
    Active Streamlit operations UI and operations services.

scripts/
    Runtime state, model adapters, risk engine, fusion, provisioning,
    demo seeding, training utilities, and tests.

model/
    Tracked model metadata/results and the local provisioning manifest.
    Frozen runtime binaries are provisioned locally and are not stored in Git.

archive/
    Intentional historical project progression, including legacy UI,
    scripts, analyses, experiments, and historical model material.

runtime/
    Local generated SQLite runtime state and operational records.
```

**Active entrypoint:** `ui/app.py`

**Historical archive application:** `archive/legacy_ui/app.py`

The archive is intentional project-history evidence and should not be deleted or treated as disposable generated code.

## Public vs Local Artifacts

Tracked and public:

- Active application source
- Tests
- `requirements.txt`
- Model metadata and evaluation results
- `model/model_manifest.json`
- `scripts/provision_models.py`
- `scripts/seed_demo_data.py`
- The intentional `archive/`

Not stored in normal Git:

- `model/*.pkl`
- `runtime/riskpulse_state.db`
- Training and research CSV datasets
- `.env` files
- Credentials and secrets
- Caches and virtual environments

The model binaries are obtained through the public GitHub Release `v1.0.0` using `scripts/provision_models.py`. The repository stores the model metadata and manifest, while the large trained model binaries remain release assets rather than Git-tracked files.

## Limitations / Prototype Notes

- RiskPulse is a prototype/buildathon system.
- Model metrics are evaluation results, not production guarantees.
- Challenge verification is represented through lifecycle events rather than a real customer identity-verification provider.
- Demo data is synthetic and deterministic.
- Runtime state is local SQLite.
- Model binaries are distributed separately through Release assets.
- Production deployment would require additional infrastructure, monitoring, model governance, security controls, data governance, and real verification and payment integrations.

## Future Production Considerations

A production deployment would need enterprise identity integration, managed and encrypted state, stronger secrets management, operational monitoring, model governance, data retention controls, privacy review, release artifact governance, and integrations with real verification and payment systems.

This repository documents the demonstrated prototype and its reproducible local evaluation path; it does not claim to provide those production controls.
