"use client";

import { useState, useEffect, useCallback } from "react";
import {
  checkHealth, predict,
  SAMPLE_NORMAL, SAMPLE_ATTACK,
  type ConnectionRecord, type PredictionResponse,
} from "@/lib/api";

// ── Feature group definitions ─────────────────────────────────────────────
const PROTOCOL_TYPES = ["tcp", "udp", "icmp"];
const SERVICES = ["http","ftp","smtp","ssh","dns","ftp_data","mtp","finger","telnet","eco_i","login","private","domain_u","imap4","Z39_50","time","daytime","bgp","ldap","ecr_i","gopher","vmnet","systat","http_443","efs","whois","irc","pop_3","netbios_ns","csnet_ns","kshell","supdup","courier","ctf","iso_tsap","link","netbios_dgm","auth","pop_2","uucp_path","sunrpc","klogin","remote_job","sql_net","hostnames","exec","discard","nntp","ntp_u","uucp","name","shell","X11","netstat","pm_dump","IRC","tim_i","printer","nnsp","other"];
const FLAGS = ["SF","S0","REJ","RSTO","RSTR","S1","S2","S3","OTH","SH"];

type Tab = "basic" | "content" | "traffic" | "host";
type HistoryItem = PredictionResponse & { seq: number; time: string };

const DEFAULT_FORM: ConnectionRecord = { ...SAMPLE_NORMAL };

export default function Home() {
  const [tab, setTab] = useState<Tab>("basic");
  const [form, setForm] = useState<ConnectionRecord>(DEFAULT_FORM);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [apiStatus, setApiStatus] = useState<"checking" | "online" | "offline">("checking");
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [totalPredictions, setTotalPredictions] = useState(0);
  const [attacksDetected, setAttacksDetected] = useState(0);

  // ── Health check ────────────────────────────────────────────────────────
  useEffect(() => {
    checkHealth()
      .then(() => setApiStatus("online"))
      .catch(() => setApiStatus("offline"));
  }, []);

  // ── Field update ─────────────────────────────────────────────────────────
  const set = useCallback((key: keyof ConnectionRecord, value: string | number) => {
    setForm(f => ({ ...f, [key]: value }));
  }, []);

  const numField = (key: keyof ConnectionRecord, label: string, step = 1, min = 0) => (
    <div className="field" key={key}>
      <label>{label}</label>
      <input
        type="number" step={step} min={min}
        value={form[key] as number}
        onChange={e => set(key, parseFloat(e.target.value) || 0)}
      />
    </div>
  );

  // ── Predict ──────────────────────────────────────────────────────────────
  const handlePredict = async () => {
    setLoading(true); setError(null);
    try {
      const res = await predict(form);
      setResult(res);
      const seq = totalPredictions + 1;
      setTotalPredictions(seq);
      if (res.is_attack) setAttacksDetected(a => a + 1);
      setHistory(h => [
        { ...res, seq, time: new Date().toLocaleTimeString() },
        ...h.slice(0, 4),
      ]);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  const isSafe = result && !result.is_attack;

  return (
    <main className="dashboard">
      {/* Header */}
      <header className="header">
        <div className="header-left">
          <div className="header-icon">🛡️</div>
          <div>
            <h1>Adaptive NIDS</h1>
            <p className="header-sub">Hybrid Zero-Day Network Intrusion Detection System</p>
          </div>
        </div>
        <div className="status-badge">
          <span className={`status-dot ${apiStatus === "online" ? "online" : apiStatus === "offline" ? "offline" : ""}`} />
          {apiStatus === "checking" ? "Connecting..." : apiStatus === "online" ? "API Online" : "API Offline"}
        </div>
      </header>

      {/* Stats Row */}
      <div className="stats-row">
        <div className="stat-card">
          <div className="stat-label">Model Architecture</div>
          <div className="stat-value blue">2-Stage</div>
          <div className="stat-sub">IF + XGBoost</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Novel Attack Recall</div>
          <div className="stat-value purple">89.4%</div>
          <div className="stat-sub">on NSL-KDD test set</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Predictions Made</div>
          <div className="stat-value green">{totalPredictions}</div>
          <div className="stat-sub">this session</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Attacks Detected</div>
          <div className="stat-value cyan">{attacksDetected}</div>
          <div className="stat-sub">{totalPredictions > 0 ? `${Math.round(attacksDetected/totalPredictions*100)}% attack rate` : "—"}</div>
        </div>
      </div>

      {/* Main Grid */}
      <div className="main-grid">
        {/* ── Left: Form ─────────────────────────────────────────── */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">Network Connection Features</span>
            <div className="tabs">
              {(["basic","content","traffic","host"] as Tab[]).map(t => (
                <button key={t} className={`tab-btn ${tab === t ? "active" : ""}`} onClick={() => setTab(t)}>
                  {t.charAt(0).toUpperCase() + t.slice(1)}
                </button>
              ))}
            </div>
          </div>
          <div className="card-body">
            {/* Quick sample buttons */}
            <div className="sample-row">
              <button className="btn-sample normal" onClick={() => { setForm(SAMPLE_NORMAL); setResult(null); setError(null); }}>
                ✓ Load Normal Sample
              </button>
              <button className="btn-sample attack" onClick={() => { setForm(SAMPLE_ATTACK); setResult(null); setError(null); }}>
                ⚠ Load Attack Sample (DoS)
              </button>
            </div>

            {/* Basic tab */}
            {tab === "basic" && (
              <div className="feature-group">
                {numField("duration","Duration (sec)")}
                <div className="field">
                  <label>Protocol Type</label>
                  <select value={form.protocol_type} onChange={e => set("protocol_type", e.target.value)}>
                    {PROTOCOL_TYPES.map(p => <option key={p}>{p}</option>)}
                  </select>
                </div>
                <div className="field">
                  <label>Service</label>
                  <select value={form.service} onChange={e => set("service", e.target.value)}>
                    {SERVICES.map(s => <option key={s}>{s}</option>)}
                  </select>
                </div>
                <div className="field">
                  <label>Flag</label>
                  <select value={form.flag} onChange={e => set("flag", e.target.value)}>
                    {FLAGS.map(f => <option key={f}>{f}</option>)}
                  </select>
                </div>
                {numField("src_bytes","Source Bytes")}
                {numField("dst_bytes","Destination Bytes")}
              </div>
            )}

            {/* Content tab */}
            {tab === "content" && (
              <div className="feature-group three-col">
                {numField("land","Land")}
                {numField("wrong_fragment","Wrong Fragment")}
                {numField("urgent","Urgent")}
                {numField("hot","Hot")}
                {numField("num_failed_logins","Failed Logins")}
                {numField("logged_in","Logged In")}
                {numField("num_compromised","Compromised")}
                {numField("root_shell","Root Shell")}
                {numField("su_attempted","SU Attempted")}
                {numField("num_root","Num Root")}
                {numField("num_file_creations","File Creations")}
                {numField("num_shells","Shells")}
                {numField("num_access_files","Access Files")}
                {numField("num_outbound_cmds","Outbound Cmds")}
                {numField("is_host_login","Host Login")}
                {numField("is_guest_login","Guest Login")}
              </div>
            )}

            {/* Traffic tab */}
            {tab === "traffic" && (
              <div className="feature-group three-col">
                {numField("count","Count")}
                {numField("srv_count","Srv Count")}
                {numField("serror_rate","SError Rate",0.01)}
                {numField("srv_serror_rate","Srv SError Rate",0.01)}
                {numField("rerror_rate","RError Rate",0.01)}
                {numField("srv_rerror_rate","Srv RError Rate",0.01)}
                {numField("same_srv_rate","Same Srv Rate",0.01)}
                {numField("diff_srv_rate","Diff Srv Rate",0.01)}
                {numField("srv_diff_host_rate","Srv Diff Host Rate",0.01)}
              </div>
            )}

            {/* Host tab */}
            {tab === "host" && (
              <div className="feature-group three-col">
                {numField("dst_host_count","DST Host Count")}
                {numField("dst_host_srv_count","DST Host Srv Count")}
                {numField("dst_host_same_srv_rate","Same Srv Rate",0.01)}
                {numField("dst_host_diff_srv_rate","Diff Srv Rate",0.01)}
                {numField("dst_host_same_src_port_rate","Same Src Port",0.01)}
                {numField("dst_host_srv_diff_host_rate","Srv Diff Host",0.01)}
                {numField("dst_host_serror_rate","SError Rate",0.01)}
                {numField("dst_host_srv_serror_rate","Srv SError Rate",0.01)}
                {numField("dst_host_rerror_rate","RError Rate",0.01)}
                {numField("dst_host_srv_rerror_rate","Srv RError Rate",0.01)}
              </div>
            )}

            <button className="btn-predict" onClick={handlePredict} disabled={loading || apiStatus !== "online"}>
              {loading ? <><span className="spinner"/>Analyzing...</> : "🔍 Analyze Connection"}
            </button>

            {error && <div className="error-banner">⚠ {error}</div>}
          </div>
        </div>

        {/* ── Right: Result + History ──────────────────────────── */}
        <div style={{display:"flex",flexDirection:"column",gap:"20px"}}>
          {/* Result card */}
          <div className="card">
            <div className="card-header">
              <span className="card-title">Prediction Result</span>
              {result && <span className={`tag ${isSafe ? "safe" : "threat"}`}>{isSafe ? "SAFE" : "THREAT"}</span>}
            </div>
            <div className="card-body">
              {!result ? (
                <div className="result-empty">
                  <div className="icon">🔮</div>
                  <p>Load a sample or fill in features, then click Analyze</p>
                </div>
              ) : (
                <div className="result-card">
                  {/* Verdict banner */}
                  <div className={`verdict-banner ${isSafe ? "safe" : "threat"}`}>
                    <div className="verdict-emoji">{isSafe ? "✅" : "🚨"}</div>
                    <div className={`verdict-label ${isSafe ? "safe" : "threat"}`}>
                      {isSafe ? "Normal Traffic" : "Attack Detected"}
                    </div>
                    <div className="verdict-family">{result.attack_family}</div>
                  </div>

                  {/* Confidence bar */}
                  <div className="confidence-bar-wrap">
                    <div className="confidence-bar-label">
                      <span>Confidence</span>
                      <span>{(result.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="confidence-bar-track">
                      <div
                        className={`confidence-bar-fill ${isSafe ? "safe" : "threat"}`}
                        style={{ width: `${result.confidence * 100}%` }}
                      />
                    </div>
                  </div>

                  {/* Metrics */}
                  <div className="metrics-grid">
                    <div className="metric-box">
                      <div className="metric-box-label">Anomaly Score</div>
                      <div className="metric-box-value" style={{color: result.anomaly_score < 0 ? "var(--danger)" : "var(--success)"}}>
                        {result.anomaly_score.toFixed(3)}
                      </div>
                    </div>
                    <div className="metric-box">
                      <div className="metric-box-label">Latency</div>
                      <div className="metric-box-value" style={{color:"var(--accent-cyan)"}}>
                        {result.processing_time_ms.toFixed(1)}ms
                      </div>
                    </div>
                  </div>

                  {/* Detail rows */}
                  <div className="detail-rows">
                    <div className="detail-row">
                      <span className="detail-row-key">Attack Type</span>
                      <span className="detail-row-val">{result.attack_type}</span>
                    </div>
                    <div className="detail-row">
                      <span className="detail-row-key">Stage 1 Decision</span>
                      <span className="detail-row-val">{result.stage1_decision}</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* History card */}
          {history.length > 0 && (
            <div className="card">
              <div className="card-header">
                <span className="card-title">Recent Predictions</span>
              </div>
              <div className="card-body" style={{paddingTop:"16px"}}>
                {history.map(h => (
                  <div className="history-row" key={h.seq}>
                    <span className="history-idx">#{h.seq}</span>
                    <span className="history-type">
                      <span className={`tag ${h.is_attack ? "threat" : "safe"}`} style={{marginRight:8}}>
                        {h.is_attack ? "ATTACK" : "NORMAL"}
                      </span>
                      {h.attack_type}
                    </span>
                    <span className="history-conf">{(h.confidence*100).toFixed(0)}%</span>
                    <span style={{fontSize:"11px",color:"var(--text-muted)",marginLeft:12}}>{h.time}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
