# LedgerLens
 
**Where Every Query Gets the Right Investigation.**
 
An agentic AI system for on-demand Anti-Money Laundering (AML) investigation. Instead of running every transaction through a fixed, expensive pipeline, LedgerLens reads the analyst's query, decides what actually needs to happen, and invokes only the relevant tools — then explains every flag in plain English.
 
---
 
## Table of Contents
 
- [Problem Statement](#problem-statement)
- [Why Traditional AML Systems Fail](#why-traditional-aml-systems-fail)
- [Our Solution Approach](#our-solution-approach)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Dataset Information](#dataset-information)
- [Data Sources](#data-sources)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Example Queries](#example-queries)
- [Team](#team)
- [License](#license)
---
 
## Problem Statement
 
### AI-Powered Suspicious Activity Detection
#### Business Summary:
Financial institutions globally are mandated by regulatory bodies (FinCEN, FATF, local authorities) to implement robust Anti-Money Laundering (AML) compliance programs. However, traditional rule-based systems generate excessive false positives, overwhelming compliance teams and increasing operational costs. Meanwhile, sophisticated money laundering techniques—including structuring , smurfing, and layering evade conventional detection methods.
The challenge is to build an intelligent, autonomous agent that can learn from transaction patterns, identify suspicious behaviours, and provide explainable risk assessments with actionable escalation recommendations. Such an agent would reduce false positives, improve detection accuracy, and enable compliance teams to focus on genuine threats rather than manual rule tuning.
### Objective:
Participants are required to design and implement an AI-powered agent that:
1.	Performs automated exploratory data analysis (EDA) on transaction and customer data to understand baseline behavior
2.	Detects anomalous transaction patterns indicative of money laundering (example - structuring/smurfing)
3.	Applies anomaly detection (e.g., any ML-based approach, rule based or Hybrid)
4.	Generates a risk score or flag per transaction/customer
5.	Provides a explanation for why a transaction is flagged as suspicious
6.	Recommends a basic escalation action (monitor / flag for review / report)

---
 
## Why Traditional AML Systems Fail
 
| Query | What a Fixed Pipeline Does (Wasteful) | What LedgerLens Does |
|---|---|---|
| "Find structuring patterns in the last 30 days" | Runs full EDA + every detector on the whole dataset | Applies time filter, runs only structuring-focused feature engineering and detection |
| "Which customers made 10+ transactions under $10,000?" | Runs ML anomaly detection unnecessarily | Runs a direct aggregation/threshold rule — no ML needed |
| "Is customer ID 4521 suspicious?" | Reprocesses the entire dataset | Performs a single-entity lookup and computes/explains risk on demand |
 
This selective, query-driven behavior — not the ML model itself — is the core engineering challenge and the core differentiator of this project.
 
---
