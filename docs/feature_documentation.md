# Feature Documentation
## Fraud Detection Pipeline — LightGBM Model
**Dataset:** 50,000 transactions | Jan–Jun 2024
**Final feature count:** 20
**Target variable:** `is_fraud` (binary: 0 = legitimate, 1 = fraud)

---

## 1. Temporal Features

| Feature | Type | Range | Description | Fraud Signal |
|---------|------|-------|-------------|--------------|
| `hour` | int | 0–23 | Hour of day the transaction occurred | Fraud peaks at 3am (39.6% rate vs 4.0% average) |
| `day_of_week` | int | 0–6 | Day encoded as 0 (Monday) to 6 (Sunday) | Monday shows highest fraud rate (4.4%) |
| `month` | int | 1–6 | Calendar month of transaction | Captures seasonal fraud drift over 181-day window |
| `is_night` | binary | 0/1 | 1 if transaction occurs between 00:00–05:00 | Night hours average 30.6% fraud rate vs 4.0% overall |
| `is_weekend` | binary | 0/1 | 1 if transaction occurs on Saturday or Sunday | Sunday is highest-volume day — unusual for a payments platform |

**Engineering note:** Extracted directly from `timestamp`. The raw `timestamp`
column is dropped after extraction.

---

## 2. Transaction Velocity Features

| Feature | Type | Range | Description | Fraud Signal |
|---------|------|-------|-------------|--------------|
| `sender_total_tx` | int | 1–17 | Total transactions sent by this sender in the dataset | High velocity may indicate automated fraud scripts or account takeover |
| `sender_unique_recv` | int | 1–n | Number of distinct receivers contacted by this sender | Fraudsters spray funds across many receivers to obscure the trail |

**Engineering note:** Computed via `groupby('sender_id')` aggregation on the
full training set. In production, these must be maintained in a live feature
store and updated incrementally.

---

## 3. Amount Features

| Feature | Type | Range | Description | Fraud Signal |
|---------|------|-------|-------------|--------------|
| `log_amount` | float | 0.95–12.24 | log(1 + amount) | Raw amount skewness of 9.11 makes it unusable directly; log normalises the distribution |
| `is_high_value` | binary | 0/1 | 1 if transaction amount exceeds KES 10,000 | Fraud rate climbs consistently above KES 10K; reaches 94% at KES 50K+ |
| `amount_vs_sender_avg` | float | 0–n | Transaction amount divided by sender's historical average | Flags anomalously large transactions relative to a specific user's own norm |

**Engineering note:** Raw `amount` is dropped after these features are computed.
`amount_vs_sender_avg` requires `sender_avg_amount` as an intermediate — this
intermediate is dropped from the final feature set.

---

## 4. Behavioural Features

| Feature | Type | Range | Description | Fraud Signal |
|---------|------|-------|-------------|--------------|
| `sender_unique_devices` | int | 1–n | Number of distinct devices used by this sender | 84.9% of fraud senders use >1 device vs 57.4% legitimate |
| `is_device_switch` | binary | 0/1 | 1 if current transaction device differs from sender's most-used device | 3rd strongest SHAP feature (mean \|SHAP\| = 1.030) — primary account takeover signal |
| `balance_drain_rate` | float | 0–1 | amount ÷ sender_balance_before, clipped to [0, 1] | Fraud transactions leave 34× more accounts below 10% balance remaining |

**Engineering note:** `is_device_switch` requires computing each sender's primary
device (most frequently used) before comparison. In production this must be
resolved from a **persistent device registry** — not recomputed from the
current batch — to avoid misclassifying legitimate users who switched devices
and to correctly flag fraud from previously unseen devices.

---

## 5. Geographic Features

| Feature | Type | Range | Description | Fraud Signal |
|---------|------|-------|-------------|--------------|
| `dist_from_nairobi` | float | 0–1,176 km | Haversine distance from Nairobi (−1.2921, 36.8219) | Top SHAP feature (mean \|SHAP\| = 2.506) — fraud median 421 km vs 248 km legitimate |
| `is_outside_kenya` | binary | 0/1 | 1 if coordinates fall outside lat [−5, 5] / lon [34, 42] | 29.3% of fraud originates outside Kenya vs 0.0% legitimate |
| `device_unique_senders` | int | 1–n | Number of distinct senders who have used this device | Baseline device risk signal — solo-device senders show higher fraud rate (5.2%) than shared |

**Engineering note:** Computed using the Haversine formula.
Raw `location_lat` and `location_lon` should be dropped after these features
are created in production to avoid redundancy. In the current pipeline they
were retained and leaked residual signal — flagged for removal in the next
iteration.

---

## 6. Balance Consistency Features

| Feature | Type | Range | Description | Fraud Signal |
|---------|------|-------|-------------|--------------|
| `sender_balance_change` | float | −n to +n | sender_balance_before − sender_balance_after | 4th strongest SHAP feature (mean \|SHAP\| = 0.882) — large deviations flag account draining |
| `receiver_balance_change` | float | −n to +n | receiver_balance_after − receiver_balance_before | Consistency check on the receiving side of the transaction |
| `balance_discrepancy` | binary | 0/1 | 1 if \|sender_balance_change − amount\| > KES 1 | Direct flag for balance manipulation — possible system-level fraud |

**Engineering note:** All four raw balance columns (`sender_balance_before`,
`sender_balance_after`, `receiver_balance_before`, `receiver_balance_after`)
are dropped after these features are computed.

---

## 7. Categorical Features

| Feature | Type | Range | Description | Fraud Signal |
|---------|------|-------|-------------|--------------|
| `transaction_type_enc` | int | 0–4 | Label-encoded transaction type | Weak standalone signal (3.7–4.5% fraud rate range) but meaningful in interactions with amount and time features |

**Encoding map:**

| Label | Code |
|-------|------|
| `send_money` | 0 |
| `pay_bill` | 1 |
| `buy_goods` | 2 |
| `withdraw` | 3 |
| `deposit` | 4 |

**Engineering note:** Fixed mapping applied consistently across train and test
sets. One-hot encoding was considered but label encoding was preferred given
LightGBM's native handling of categorical integers.

---

## 8. Dropped Columns

The following raw columns are removed before model training:

| Column | Reason |
|--------|--------|
| `timestamp` | Superseded by `hour`, `day_of_week`, `month`, `is_night`, `is_weekend` |
| `sender_id` | Identifier — not a generalizable feature |
| `receiver_id` | Identifier — not a generalizable feature |
| `device_id` | Identifier — superseded by `is_device_switch`, `sender_unique_devices` |
| `transaction_type` | Superseded by `transaction_type_enc` |
| `location_lat` | Superseded by `dist_from_nairobi`, `is_outside_kenya` |
| `location_lon` | Superseded by `dist_from_nairobi`, `is_outside_kenya` |
| `amount` | Superseded by `log_amount`, `is_high_value`, `amount_vs_sender_avg` |
| `sender_balance_before` | Superseded by `sender_balance_change`, `balance_drain_rate` |
| `sender_balance_after` | Superseded by `sender_balance_change`, `balance_drain_rate` |
| `receiver_balance_before` | Superseded by `receiver_balance_change` |
| `receiver_balance_after` | Superseded by `receiver_balance_change` |

---

## 9. Feature Importance Summary (SHAP)

| Rank | Feature | Mean \|SHAP\| | Signal Category |
|------|---------|--------------|-----------------|
| 1 | `dist_from_nairobi` | 2.506 | Geographic |
| 2 | `hour` | 1.699 | Temporal |
| 3 | `is_device_switch` | 1.030 | Behavioural |
| 4 | `sender_balance_change` | 0.882 | Balance |
| 5 | `is_outside_kenya` | 0.543 | Geographic |
| 6 | `month` | 0.528 | Temporal |
| 7 | `sender_unique_devices` | 0.499 | Behavioural |

---

## 10. Production Notes

- **Velocity features** (`sender_total_tx`, `sender_unique_recv`) must be
  maintained in a live feature store updated after every transaction.
- **Device features** (`is_device_switch`, `sender_unique_devices`) require a
  persistent device registry keyed by `sender_id`. Recomputing from batch data
  will cause both false positives (legitimate users flagged after switching phones)
  and false negatives (known-bad devices appearing new).
- **Geographic features** are computed at inference time from raw lat/lon —
  ensure the Haversine function is available in the serving environment.
- **Threshold:** Default decision threshold is 0.5. Recommend calibrating to
  0.3–0.4 in production if recall is the operational priority.