# NSL-KDD Feature-by-Feature Analysis Report

This document provides a highly detailed feature dictionary and technical analysis of all **41 features** in the NSL-KDD dataset. It evaluates each variable's data type, expected range, security significance, preprocessing requirements (encoding and scaling), and contribution to model convergence.

---

## 1. Complete Feature Dictionary

### I. Basic Connection Features
These features capture the structural characteristics of packet headers without reading the payload.

| Feature Name | Data Type | Description | Expected Range | Security Importance | Encoding? | Scaling? |
|--------------|-----------|-------------|----------------|---------------------|-----------|----------|
| `duration` | Continuous | Length of the connection in seconds | $[0, \infty)$ | Critical for identifying prolonged persistent connections (like SSH tunnels) vs fast probes. | No | Yes (Log) |
| `protocol_type` | Categorical | Transmission protocol | `tcp`, `udp`, `icmp` | High. Certain attacks are strictly tied to a protocol (e.g., Neptune requires TCP SYN; Smurf uses ICMP). | Yes (OHE) | No |
| `service` | Categorical | Destination service | `http`, `smtp`, `dns` etc. (70+) | High. Probes scan widely; DoS often targets `http`; R2L targets interactive ports (`ftp`, `telnet`). | Yes (Target) | No |
| `flag` | Categorical | Connection status flags | `SF`, `S0`, `REJ`, etc. (11 states) | Critical. Highlights anomalous connections (e.g., `S0` indicates half-open TCP SYN floods). | Yes (OHE) | No |
| `src_bytes` | Continuous | Bytes from source to dest | $[0, \infty)$ | Critical. Massive values show exfiltration, high-volume DoS, or data transfers. | No | Yes (Log) |
| `dst_bytes` | Continuous | Bytes from dest to source | $[0, \infty)$ | Critical. Large sizes represent data downloads, HTTP responses, or buffer payloads. | No | Yes (Log) |
| `land` | Binary | 1 if src & dst IP/port match; 0 otherwise | `0` or `1` | High. Extremely rare, indicates a Land Loopback denial-of-service attack attempt. | No | No |
| `wrong_fragment` | Continuous | Number of wrong fragments | $[0, \infty)$ | High. Non-zero values indicate fragmentation exploits (e.g., Teardrop, Ping of Death). | No | No |
| `urgent` | Continuous | Number of urgent packets | $[0, \infty)$ | Medium. Out-of-order execution, often used in stealth scanning or payload exploitation. | No | No |

---

### II. Content Features
These features inspect the payload, exposing unauthorized actions, logins, and privilege escalations.

| Feature Name | Data Type | Description | Expected Range | Security Importance | Encoding? | Scaling? |
|--------------|-----------|-------------|----------------|---------------------|-----------|----------|
| `hot` | Continuous | Count of hot indicators | $[0, \infty)$ | High. Triggers on actions like executing programs, accessing system roots, or unauthorized directories. | No | Yes (Robust) |
| `num_failed_logins`| Continuous | Number of failed login attempts | $[0, \infty)$ | Critical. Directly exposes brute-force or dictionary login attacks. | No | No |
| `logged_in` | Binary | 1 if login succeeds; 0 otherwise | `0` or `1` | High. Helps differentiate between successful breaches and failed login attacks. | No | No |
| `num_compromised` | Continuous | Count of compromised indicators | $[0, \infty)$ | High. Escalations, system crashes, or file overrides. | No | Yes (Robust) |
| `root_shell` | Binary | 1 if root shell acquired; 0 otherwise | `0` or `1` | Critical. The primary indicator of a successful User-to-Root (U2R) exploit. | No | No |
| `su_attempted` | Categorical | Status of su root command | `0`, `1`, `2` | High. Captures attempts to escalate standard terminal sessions to admin level. | Yes (OHE) | No |
| `num_root` | Continuous | Count of root accesses | $[0, \infty)$ | High. Audit trail of admin actions. | No | Yes (Robust) |
| `num_file_creations`| Continuous | Count of file creation actions | $[0, \infty)$ | High. Essential to detect rootkit deployment or compiler runs. | No | Yes (Robust) |
| `num_shells` | Continuous | Count of shell prompts spawned | $[0, \infty)$ | Critical. Spawning shells within non-interactive services signals a remote command exploit. | No | No |
| `num_access_files` | Continuous | Count of files modified/read | $[0, \infty)$ | High. Focuses on system security configuration adjustments (e.g., `.rhosts`). | No | No |
| `num_outbound_cmds`| Continuous | FTP outbound commands count | Always `0` | None. Dead column in NSL-KDD. **Should be dropped.** | No | No |
| `is_hot_login` | Binary | 1 if hot login; 0 otherwise | `0` or `1` | Low. Triggers on specific system logins. | No | No |
| `is_guest_login` | Binary | 1 if guest login; 0 otherwise | `0` or `1` | High. Useful to track guest privileges abused during R2L. | No | No |

---

### III. Time-Based Traffic Features
Aggregated features using a 2-second time window. Excellent for rapid volumetric changes.

| Feature Name | Data Type | Description | Expected Range | Security Importance | Encoding? | Scaling? |
|--------------|-----------|-------------|----------------|---------------------|-----------|----------|
| `count` | Continuous | Connection count to same host | $[0, \infty)$ | Critical. Floods and rapid port scans generate high connection density in a short window. | No | Yes (Robust) |
| `srv_count` | Continuous | Connection count to same service | $[0, \infty)$ | Critical. Identifies service-focused floods (e.g., HTTP flood, SYN flood). | No | Yes (Robust) |
| `serror_rate` | Continuous | % of SYN error connections | $[0.0, 1.0]$ | Critical. Flags Neptune or SYN scanning activities. | No | No |
| `srv_serror_rate` | Continuous | % of same-service SYN errors | $[0.0, 1.0]$ | Critical. Focuses on service-specific SYN errors. | No | No |
| `rerror_rate` | Continuous | % of REJ error connections | $[0.0, 1.0]$ | Critical. Identifies active port scanning hitting closed ports. | No | No |
| `srv_rerror_rate` | Continuous | % of same-service REJ errors | $[0.0, 1.0]$ | Critical. Identifies scanning targeting specific closed services. | No | No |
| `same_srv_rate` | Continuous | % of same-service connections | $[0.0, 1.0]$ | High. Helps detect uniform flood traffic. | No | No |
| `diff_srv_rate` | Continuous | % of different-service connections| $[0.0, 1.0]$ | Critical. High rates reveal scanning across multiple ports. | No | No |
| `srv_diff_host_rate`| Continuous | % of different-host connections | $[0.0, 1.0]$ | High. Identifies distributed host attacks. | No | No |

---

### IV. Host-Based Traffic Features
Analyzes the last 100 connections to the same destination IP to identify stealthy, slow-moving attacks.

| Feature Name | Data Type | Description | Expected Range | Security Importance | Encoding? | Scaling? |
|--------------|-----------|-------------|----------------|---------------------|-----------|----------|
| `dst_host_count` | Continuous | IP-to-host connection count | $[0, 255]$ | High. Tracks long-term target focus. | No | Yes (MinMax) |
| `dst_host_srv_count`| Continuous | Host service connection count | $[0, 255]$ | High. Tracks long-term target service usage. | No | Yes (MinMax) |
| `dst_host_same_srv_rate`| Continuous | % same-service to host | $[0.0, 1.0]$ | High. Recognizes targeted service floods. | No | No |
| `dst_host_diff_srv_rate`| Continuous | % diff-service to host | $[0.0, 1.0]$ | Critical. Tracks slow port sweeps targeting a single host. | No | No |
| `dst_host_same_src_port_rate`| Continuous | % same source port to host | $[0.0, 1.0]$ | Critical. Detects scans generating traffic from a single port. | No | No |
| `dst_host_srv_diff_host_rate`| Continuous | % diff-host same-service | $[0.0, 1.0]$ | High. Highlights distributed activity. | No | No |
| `dst_host_serror_rate`| Continuous | % host-level SYN errors | $[0.0, 1.0]$ | Critical. Detects stealthy SYN floods. | No | No |
| `dst_host_srv_serror_rate`| Continuous | % host service SYN errors | $[0.0, 1.0]$ | Critical. Detects stealthy service-focused SYN floods. | No | No |
| `dst_host_rerror_rate`| Continuous | % host-level REJ errors | $[0.0, 1.0]$ | Critical. Detects host sweeps on closed ports. | No | No |
| `dst_host_srv_rerror_rate`| Continuous | % host service REJ errors | $[0.0, 1.0]$ | Critical. Detects focused scans hitting closed services. | No | No |

---

## 2. Technical Feature Insights

### A. Highly Informative Features
These variables provide high informational entropy and act as the core indicators for classifications:
1.  **`flag`**: The status of connection states (e.g., `S0`, `REJ`, `SF`) acts as a primary sensor for TCP protocol abuse.
2.  **`src_bytes` & `dst_bytes`**: Crucial for tracking the scale of data transfers. Spikes indicate volume floods or file extractions; near-zero values with long durations indicate idle tunnels.
3.  **`diff_srv_rate` & `dst_host_diff_srv_rate`**: The leading indicators for network scanning and active probes.
4.  **`serror_rate` & `dst_host_serror_rate`**: Standard indicators for SYN floods and half-open TCP attacks.

### B. Highly Redundant / Multi-Collinear Features
Multiple rate features track identical events across different aggregation windows (2-second vs. 100-connection windows). This introduces redundancy:
-   **SYN Error Multiplicity**: `serror_rate`, `srv_serror_rate`, `dst_host_serror_rate`, and `dst_host_srv_serror_rate` share a correlation coefficient $> 0.96$ in active floods.
-   **REJ Error Multiplicity**: `rerror_rate`, `srv_rerror_rate`, `dst_host_rerror_rate`, and `dst_host_srv_rerror_rate` also show high multi-collinearity.
-   *Recommendation*: Drop highly correlated intermediate rates or rely on tree-based models (XGBoost) which are robust to multicollinearity, rather than linear models.

### C. Candidate Features for Advanced Feature Engineering
In future development phases, we can engineer new features to improve classifier capabilities:
1.  **Bytes Ratio (`src_bytes` / (`dst_bytes` + 1))**: Highlights unidirectional packet streams, a hallmark of DoS attacks and bulk uploads.
2.  **Failed-to-Total Connection Ratio (`serror_rate` + `rerror_rate`)**: Represents a general "Anomalous Connection Index" combining both refused and timed-out connections.
3.  **Log-Scaled Payload Density (`log(src_bytes + 1) * logged_in`)**: Helps separate large benign transfers from massive payload exploits in authenticated sessions.
