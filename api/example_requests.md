# LedgerLens API — Example Requests

Base URL: `http://localhost:8000`

Run the server with:
```bash
uvicorn api.main:app --reload --port 8000
```

---

## GET /health

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "ok",
  "dataset_loaded": true,
  "dataset_rows": 50001,
  "dataset_columns": ["Time","Date","Sender_account","Receiver_account","Amount","Payment_currency","Received_currency","Sender_bank_location","Receiver_bank_location","Payment_type","Is_laundering","Laundering_type","Timestamp"]
}
```

---

## POST /investigate

### Query 1 — Aggregation (customers with 10+ transactions under $10k)

```bash
curl -X POST http://localhost:8000/investigate \
  -H "Content-Type: application/json" \
  -d '{"query": "Which customers made 10+ transactions under $10,000?"}'
```

### Query 2 — Entity Lookup (single customer suspicion check)

Replace `8724731955` with any real Sender_account from the dataset.

```bash
curl -X POST http://localhost:8000/investigate \
  -H "Content-Type: application/json" \
  -d '{"query": "Is customer ID 8724731955 suspicious?"}'
```

### Query 3 — Broad Exploration (overview of transaction patterns)

```bash
curl -X POST http://localhost:8000/investigate \
  -H "Content-Type: application/json" \
  -d '{"query": "Show me an overview of transaction patterns"}'
```

---

## Error Cases

### 400 — Empty query
```bash
curl -X POST http://localhost:8000/investigate \
  -H "Content-Type: application/json" \
  -d '{"query": ""}'
```

### 404 — Entity not found
```bash
curl -X POST http://localhost:8000/investigate \
  -H "Content-Type: application/json" \
  -d '{"query": "Is customer ID 0000000000 suspicious?"}'
```

---

> **Note on `calibration_warning`**: Until the `feat/anomaly-engine` branch is merged,
> `anomaly_node` returns stub scores. The API surfaces this explicitly via the
> `calibration_warning` field so demo reviewers are not misled.
