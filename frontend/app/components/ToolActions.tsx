"use client";

import { useState, useEffect, useRef } from "react";
import { useCopilotAction } from "@copilotkit/react-core";

/* ─────────────────────────────────────────────────────────── */
/* Module-level SSE store — persists across React re-renders   */
/* ─────────────────────────────────────────────────────────── */
const _sseStore = {
  liveRounds: {} as Record<string, any[]>,
  liveResults: [] as any[],
};

function resetSseStore() {
  _sseStore.liveRounds = {};
  _sseStore.liveResults = [];
}

if (typeof window !== "undefined") {
  window.addEventListener("ura-search-event", (e: Event) => {
    const event = (e as CustomEvent).detail;
    if (event.type === "round") {
      const key = `${event.query_idx}-${event.agent}`;
      _sseStore.liveRounds = {
        ..._sseStore.liveRounds,
        [key]: [...(_sseStore.liveRounds[key] ?? []), event],
      };
    } else if (event.type === "query_done") {
      const next = [..._sseStore.liveResults];
      next[event.query_idx] = event;
      _sseStore.liveResults = next;
    }
    window.dispatchEvent(new CustomEvent("ura-render-tick"));
  });
}

/* ─────────────────────────────────────────────────────────── */
/* Shared styles                                               */
/* ─────────────────────────────────────────────────────────── */
const outerCard: React.CSSProperties = {
  background: "#10121c",
  border: "1px solid #252840",
  borderRadius: 14,
  padding: "16px 18px",
  fontSize: 13,
  lineHeight: 1.6,
  width: "100%",
  maxWidth: 640,
};

const sectionLabel = (color: string): React.CSSProperties => ({
  fontSize: 10,
  fontWeight: 700,
  letterSpacing: "0.1em",
  textTransform: "uppercase" as const,
  color,
  marginBottom: 6,
});

const pill = (color: string): React.CSSProperties => ({
  display: "inline-flex",
  alignItems: "center",
  gap: 4,
  background: color + "22",
  border: `1px solid ${color}44`,
  color,
  borderRadius: 6,
  padding: "2px 8px",
  fontSize: 10,
  fontWeight: 700,
  letterSpacing: "0.06em",
  textTransform: "uppercase" as const,
  whiteSpace: "nowrap" as const,
});

const tinyCard: React.CSSProperties = {
  background: "#1a1d2e",
  border: "1px solid #2a2d42",
  borderRadius: 8,
  padding: "8px 10px",
  marginBottom: 6,
};

const KEYFRAMES = `
  @keyframes ura-shimmer { 0%{background-position:120% 0} 100%{background-position:-120% 0} }
  @keyframes ura-spin    { from{transform:rotate(0deg)}   to{transform:rotate(360deg)} }
`;

/* Score color — green ≥0.7, amber 0.4–0.7, red <0.4 */
function scoreColor(score: number): string {
  if (score >= 0.7) return "#22c55e";
  if (score >= 0.4) return "#f59e0b";
  return "#ef4444";
}

/* Robust check: the backend uses several "no relevant" message variants */
function isNoResult(answer: string | undefined | null): boolean {
  if (!answer) return true;
  return /no relevant/i.test(answer);
}

/* PATH C signal: agent found related content but couldn't directly answer */
function isRelatedContent(answer: string | undefined | null): boolean {
  if (!answer) return false;
  return /related .* content exists/i.test(answer);
}

/* Combined: answer is negative (either no-result or related-but-can't-answer) */
function isNegativeAnswer(answer: string | undefined | null): boolean {
  return isNoResult(answer) || isRelatedContent(answer);
}

function Spinner({ size = 12, color = "currentColor" }: { size?: number; color?: string }) {
  return (
    <>
      <style>{KEYFRAMES}</style>
      <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
        style={{ animation: "ura-spin 0.9s linear infinite", flexShrink: 0 }}>
        <circle cx="12" cy="12" r="10" stroke={color} strokeWidth="3"
          strokeLinecap="round" strokeDasharray="31.4 31.4" />
      </svg>
    </>
  );
}

function Shimmer({ width, color, highlight, delay = 0 }: {
  width: string; color: string; highlight: string; delay?: number;
}) {
  return (
    <div style={{
      height: 28, borderRadius: 7, marginBottom: 5,
      background: `linear-gradient(90deg,${color} 25%,${highlight} 50%,${color} 75%)`,
      backgroundSize: "300% 100%",
      animation: `ura-shimmer 1.5s ease-in-out ${delay}s infinite`,
      width,
    }} />
  );
}

/* ─────────────────────────────────────────────────────────── */
/* Pipeline stepper — visual 4-step phase tracker              */
/* ─────────────────────────────────────────────────────────── */
/* Mini SVG icons for steps — simple, consistent, render everywhere */
function StepIcon({ type, size = 13, color }: { type: string; size?: number; color: string }) {
  const props = { width: size, height: size, viewBox: "0 0 24 24", fill: "none", stroke: color, strokeWidth: 2.2, strokeLinecap: "round" as const, strokeLinejoin: "round" as const };
  switch (type) {
    case "analyze":
      return <svg {...props}><circle cx="11" cy="11" r="7" /><line x1="16.5" y1="16.5" x2="21" y2="21" /></svg>;
    case "search":
      return <svg {...props}><path d="M4 6h16M4 12h10M4 18h6" /></svg>;
    case "synthesize":
      return <svg {...props}><path d="M6 3v6l6 3-6 3v6" /><path d="M18 3v6l-6 3 6 3v6" /></svg>;
    case "done":
      return <svg {...props} stroke="none" fill={color}><path d="M9 16.2L4.8 12l-1.4 1.4L9 19 21 7l-1.4-1.4L9 16.2z" /></svg>;
    default:
      return null;
  }
}

const STEPS = [
  { key: "starting",      label: "Analyze",     iconType: "analyze"    },
  { key: "retrieving",    label: "Search",      iconType: "search"     },
  { key: "synthesizing",  label: "Synthesize",  iconType: "synthesize" },
  { key: "complete",      label: "Done",        iconType: "done"       },
] as const;

function PipelineStepper({ phase, doneCount, totalQueries }: {
  phase: Phase; doneCount: number; totalQueries: number;
}) {
  const phaseOrder: Record<Phase, number> = {
    starting: 0, retrieving: 1, synthesizing: 2, complete: 3, error: -1,
  };
  const currentIdx = phaseOrder[phase] ?? -1;

  // Progress percentage for the bar
  let pct = 0;
  if (phase === "starting") pct = 8;
  else if (phase === "retrieving") pct = totalQueries > 0 ? 15 + (doneCount / totalQueries) * 55 : 20;
  else if (phase === "synthesizing") pct = 80;
  else if (phase === "complete") pct = 100;

  return (
    <div style={{ marginBottom: 14 }}>
      {/* Step circles + connector line */}
      <div style={{ display: "flex", alignItems: "center", gap: 0, position: "relative", padding: "0 4px" }}>
        {STEPS.map(({ key, label, iconType }, i) => {
          const isDone = currentIdx > i || phase === "complete";
          const isCurrent = currentIdx === i && phase !== "error";
          const color = isDone ? "#22c55e" : isCurrent ? "#818cf8" : "#334155";
          const bg = isDone ? "#22c55e22" : isCurrent ? "#6366f122" : "#1a1d2e";
          const borderColor = isDone ? "#22c55e55" : isCurrent ? "#6366f155" : "#2d3148";
          return (
            <div key={key} style={{ display: "flex", alignItems: "center", flex: i < STEPS.length - 1 ? 1 : "none" }}>
              {/* Step node */}
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4, flexShrink: 0 }}>
                <div style={{
                  width: 28, height: 28, borderRadius: "50%",
                  background: bg, border: `1.5px solid ${borderColor}`,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  transition: "all 0.3s ease",
                  ...(isCurrent ? { boxShadow: `0 0 8px ${color}44` } : {}),
                }}>
                  {isCurrent && phase !== "complete" ? <Spinner size={12} color={color} /> : <StepIcon type={iconType} color={color} />}
                </div>
                <span style={{
                  fontSize: 9, fontWeight: 700, color: isCurrent ? "#c4b5fd" : isDone ? "#4ade80" : "#475569",
                  letterSpacing: "0.05em", textTransform: "uppercase",
                  transition: "color 0.3s ease",
                }}>
                  {label}
                  {isCurrent && phase === "retrieving" && totalQueries > 0
                    ? ` ${doneCount}/${totalQueries}`
                    : ""}
                </span>
              </div>
              {/* Connector line */}
              {i < STEPS.length - 1 && (
                <div style={{
                  flex: 1, height: 1.5, margin: "0 6px", marginBottom: 18,
                  background: currentIdx > i ? "#22c55e55" : "#2d3148",
                  transition: "background 0.3s ease",
                }} />
              )}
            </div>
          );
        })}
      </div>

      {/* Thin progress bar */}
      <div style={{
        height: 3, borderRadius: 2, background: "#1a1d2e",
        overflow: "hidden", marginTop: 8,
      }}>
        <div style={{
          height: "100%", borderRadius: 2,
          width: `${pct}%`,
          background: phase === "complete"
            ? "linear-gradient(90deg, #22c55e, #4ade80)"
            : phase === "error"
            ? "#ef4444"
            : "linear-gradient(90deg, #6366f1, #818cf8)",
          transition: "width 0.5s ease",
        }} />
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────── */
/* Elapsed timer hook                                          */
/* ─────────────────────────────────────────────────────────── */
function useElapsed(isRunning: boolean) {
  const [secs, setSecs] = useState(0);
  useEffect(() => {
    if (!isRunning) return;
    setSecs(0);
    const id = setInterval(() => setSecs(s => s + 1), 1000);
    return () => clearInterval(id);
  }, [isRunning]);
  return secs;
}

/* ─────────────────────────────────────────────────────────── */
/* Agent round card                                            */
/* ─────────────────────────────────────────────────────────── */
function RoundCard({ round, color, accent }: { round: any; color: string; accent: string }) {
  return (
    <div style={{ ...tinyCard, borderLeft: `2px solid ${accent}55`, marginBottom: 5 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 3 }}>
        <span style={{ fontSize: 9, fontWeight: 700, color, background: `${accent}22`, borderRadius: 4, padding: "1px 5px" }}>
          Round {round.round}{round.round > 1 ? " · reformulation" : ""}
        </span>
        <span style={{ fontSize: 10, color: "#475569" }}>{round.num_results} results</span>
      </div>
      <div style={{ color: "#94a3b8", fontSize: 11, fontStyle: "italic" }}>"{round.tool_query}"</div>
      {round.reasoning && (
        <div style={{ color: "#64748b", fontSize: 11, marginTop: 3, borderTop: "1px solid #252840", paddingTop: 3 }}>
          {round.reasoning}
        </div>
      )}
    </div>
  );
}

/* ─────────────────────────────────────────────────────────── */
/* Sub-query lane — live while retrieval in progress           */
/* ─────────────────────────────────────────────────────────── */
function LiveQueryLane({ idx, query, faqRounds, docRounds }: {
  idx: number; query: string; faqRounds: any[]; docRounds: any[];
}) {
  return (
    <div style={{ background: "#13162a", border: "1px solid #2a2d44", borderRadius: 10, padding: "12px 14px", marginBottom: 10 }}>
      <style>{`
        @keyframes ura-pulse-opacity { 0%,100%{opacity:.45} 50%{opacity:1} }
        @keyframes ura-pulse-border  { 0%,100%{border-color:#6366f122} 50%{border-color:#6366f155} }
        @keyframes ura-agent-blink   { 0%,100%{opacity:.5} 50%{opacity:1} }
      `}</style>

      {/* Sub-query header */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
        <div style={{
          width: 22, height: 22, borderRadius: "50%",
          background: "linear-gradient(135deg,#6366f1,#8b5cf6)",
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 10, fontWeight: 700, color: "#fff", flexShrink: 0,
        }}>{idx + 1}</div>
        <span style={{ color: "#c4b5fd", fontSize: 13, fontWeight: 500 }}>{query}</span>
        <span style={{ marginLeft: "auto" }}><Spinner size={12} color="#818cf8" /></span>
      </div>

      {/* ── Parallel agent columns ─────────────────────────────────── */}
      <div style={{ display: "flex", gap: 10 }}>
        {/* FAQ Agent */}
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ ...sectionLabel("#a78bfa"), display: "flex", alignItems: "center", gap: 6, marginBottom: 6 }}>
            <Spinner size={10} color="#a78bfa" />
            <span style={{ animation: "ura-agent-blink 2s ease-in-out infinite" }}>FAQ Search Agent</span>
          </div>
          {faqRounds.length === 0 ? (
            [90, 70, 50].map((w, i) => <Shimmer key={i} width={`${w}%`} color="#1a1d2e" highlight="#2d3158" delay={i * 0.2} />)
          ) : (
            <>
              {faqRounds.map((r: any, i: number) => <RoundCard key={i} round={r} color="#a78bfa" accent="#7c3aed" />)}
              <div style={{
                display: "flex", alignItems: "center", gap: 6,
                color: "#6366f1", fontSize: 11, padding: "4px 8px", marginTop: 4,
                background: "#1a1d2e", borderRadius: 5,
                animation: "ura-pulse-opacity 1.6s ease-in-out infinite",
              }}>
                <Spinner size={10} color="#818cf8" /> searching next round…
              </div>
            </>
          )}
        </div>

        {/* Doc Agent */}
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ ...sectionLabel("#38bdf8"), display: "flex", alignItems: "center", gap: 6, marginBottom: 6 }}>
            <Spinner size={10} color="#38bdf8" />
            <span style={{ animation: "ura-agent-blink 2s ease-in-out infinite" }}>Document Search Agent</span>
          </div>
          {docRounds.length === 0 ? (
            [70, 55, 38].map((w, i) => <Shimmer key={i} width={`${w}%`} color="#0f1a2e" highlight="#1a3050" delay={i * 0.2} />)
          ) : (
            <>
              {docRounds.map((r: any, i: number) => <RoundCard key={i} round={r} color="#38bdf8" accent="#0369a1" />)}
              <div style={{
                display: "flex", alignItems: "center", gap: 6,
                color: "#0ea5e9", fontSize: 11, padding: "4px 8px", marginTop: 4,
                background: "#0f1a2e", borderRadius: 5,
                animation: "ura-pulse-opacity 1.6s ease-in-out infinite",
              }}>
                <Spinner size={10} color="#38bdf8" /> searching next round…
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────── */
/* Sub-query lane — final, after sub-query fully done          */
/* ─────────────────────────────────────────────────────────── */
function DoneQueryLane({ idx, queryResult }: {
  idx: number;
  queryResult: {
    query: string;
    faq_answer?: string;
    doc_answer?: string;
    faq_results: any[];
    doc_results: any[];
    faq_attempts?: number;
    doc_attempts?: number;
    faq_steps?: Array<{ round: number; tool_query: string; num_results: number; reasoning: string }>;
    doc_steps?: Array<{ round: number; tool_query: string; num_results: number; reasoning: string }>;
  };
}) {
  const { query, faq_answer, doc_answer, faq_results = [], doc_results = [],
    faq_attempts, doc_attempts, faq_steps = [], doc_steps = [] } = queryResult;
  const faqHasAnswer = !isNegativeAnswer(faq_answer);
  const docHasAnswer = !isNegativeAnswer(doc_answer);

  return (
    <div style={{ background: "#13162a", border: "1px solid #2a2d44", borderRadius: 10, padding: "12px 14px", marginBottom: 10 }}>
      {/* Header — no query text here (already in Decomposed Queries list above) */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
        <div style={{
          width: 22, height: 22, borderRadius: "50%",
          background: "linear-gradient(135deg,#6366f1,#8b5cf6)",
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 10, fontWeight: 700, color: "#fff", flexShrink: 0,
        }}>{idx + 1}</div>
        <span style={{ color: "#94a3b8", fontSize: 13, flex: 1, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" as const }}>{query}</span>
        <div style={{ display: "flex", gap: 5, flexShrink: 0 }}>
          <span style={pill("#a78bfa")}>{faq_results.length} FAQs</span>
          <span style={pill("#38bdf8")}>{doc_results.length} Docs</span>
          {(faq_steps.length + doc_steps.length) > 2 && (
            <span style={pill("#64748b")}>{faq_steps.length + doc_steps.length} rounds</span>
          )}
        </div>
      </div>

      {/* Grounded answers */}
      <div style={{ display: "flex", gap: 10, marginBottom: 10 }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={sectionLabel("#a78bfa")}>FAQ Search Agent</div>
          <div style={{
            ...tinyCard,
            background: faqHasAnswer ? "#1e1b38" : "#171a28",
            border: `1px solid ${faqHasAnswer ? "#7c3aed44" : "#2a2d42"}`,
            color: faqHasAnswer ? "#c4b5fd" : "#475569",
            fontSize: 12, whiteSpace: "pre-wrap" as const,
          }}>{faq_answer ?? "—"}</div>
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={sectionLabel("#38bdf8")}>Document Search Agent</div>
          <div style={{
            ...tinyCard,
            background: docHasAnswer ? "#0c1f38" : "#171a28",
            border: `1px solid ${docHasAnswer ? "#0369a144" : "#2a2d42"}`,
            color: docHasAnswer ? "#bae6fd" : "#475569",
            fontSize: 12, whiteSpace: "pre-wrap" as const,
          }}>{doc_answer ?? "—"}</div>
        </div>
      </div>

      {/* Agent trace */}
      {(faq_steps.length > 0 || doc_steps.length > 0) && (
        <details style={{ cursor: "pointer", marginBottom: 8 }}>
          <summary style={{ color: "#475569", fontSize: 11, userSelect: "none" as const, marginBottom: 6 }}>
            Agent trace ({faq_steps.length + doc_steps.length} rounds)
          </summary>
          <div style={{ display: "flex", gap: 10, marginTop: 6 }}>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={sectionLabel("#a78bfa")}>FAQ Search Agent rounds</div>
              {faq_steps.map((s, i) => (
                <div key={i} style={{ ...tinyCard, borderLeft: "2px solid #7c3aed55" }}>
                  <div style={{ display: "flex", gap: 6, marginBottom: 3 }}>
                    <span style={{ fontSize: 9, fontWeight: 700, color: "#a78bfa", background: "#7c3aed22", borderRadius: 4, padding: "1px 5px" }}>Round {s.round}</span>
                    <span style={{ fontSize: 10, color: "#475569" }}>{s.num_results} results</span>
                  </div>
                  <div style={{ color: "#94a3b8", fontSize: 11, fontStyle: "italic" }}>"{s.tool_query}"</div>
                  {s.reasoning && <div style={{ color: "#64748b", fontSize: 11, marginTop: 3, borderTop: "1px solid #252840", paddingTop: 3 }}>{s.reasoning}</div>}
                </div>
              ))}
            </div>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={sectionLabel("#38bdf8")}>Document Search Agent rounds</div>
              {doc_steps.map((s, i) => (
                <div key={i} style={{ ...tinyCard, borderLeft: "2px solid #0369a155" }}>
                  <div style={{ display: "flex", gap: 6, marginBottom: 3 }}>
                    <span style={{ fontSize: 9, fontWeight: 700, color: "#38bdf8", background: "#0369a122", borderRadius: 4, padding: "1px 5px" }}>Round {s.round}</span>
                    <span style={{ fontSize: 10, color: "#475569" }}>{s.num_results} results</span>
                  </div>
                  <div style={{ color: "#94a3b8", fontSize: 11, fontStyle: "italic" }}>"{s.tool_query}"</div>
                  {s.reasoning && <div style={{ color: "#64748b", fontSize: 11, marginTop: 3, borderTop: "1px solid #252840", paddingTop: 3 }}>{s.reasoning}</div>}
                </div>
              ))}
            </div>
          </div>
        </details>
      )}

      {/* Raw retrieved items */}
      <details style={{ cursor: "pointer" }}>
        <summary style={{ color: "#475569", fontSize: 11, userSelect: "none" as const, marginBottom: 6 }}>
          Raw retrieved items ({faq_results.length + doc_results.length})
        </summary>
        <div style={{ display: "flex", gap: 10, marginTop: 6 }}>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={sectionLabel("#a78bfa")}>FAQ entries</div>
            {faq_results.length === 0
              ? <div style={{ ...tinyCard, color: "#475569", fontSize: 12 }}>No FAQs found</div>
              : faq_results.map((r: any, i: number) => (
                <div key={i} style={tinyCard}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                    <span style={{ fontSize: 10, color: "#7c3aed", fontWeight: 700 }}>Q{i + 1}</span>
                    {r.score != null && <span style={{ fontSize: 10, color: scoreColor(r.score), fontWeight: 600 }}>↑{Number(r.score).toFixed(3)}</span>}
                  </div>
                  <div style={{ color: "#94a3b8", fontSize: 12, marginBottom: 2 }}>{r.question}</div>
                  <div style={{ color: "#cbd5e1", fontSize: 12 }}>{r.answer}</div>
                </div>
              ))}
          </div>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={sectionLabel("#38bdf8")}>Document passages</div>
            {doc_results.length === 0
              ? <div style={{ ...tinyCard, color: "#475569", fontSize: 12 }}>No documents found</div>
              : doc_results.map((r: any, i: number) => (
                <div key={i} style={tinyCard}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                    <span style={{ fontSize: 10, color: "#0369a1", fontWeight: 700 }}>P{i + 1}</span>
                    {r.score != null && <span style={{ fontSize: 10, color: scoreColor(r.score), fontWeight: 600 }}>↑{Number(r.score).toFixed(3)}</span>}
                  </div>
                  <div style={{ color: "#cbd5e1", fontSize: 12, lineHeight: 1.5 }}>{r.chunk}</div>
                </div>
              ))}
          </div>
        </div>
      </details>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────── */
/* Phase system — maps pipeline state → header content         */
/*                                                             */
/* From the paper (Algorithm 1 + Figure 1):                   */
/*  1. "starting"    — Manager Agent initializing agent teams  */
/*  2. "retrieving"  — FAQ ∥ Doc agents running concurrently   */
/*  3. "synthesizing"— Manager Agent aggregating E = ∪Eᵢ       */
/*  4. "complete"    — handler returned, LLM writing answer    */
/* ─────────────────────────────────────────────────────────── */
type Phase = "starting" | "retrieving" | "synthesizing" | "complete" | "error";

/* ─────────────────────────────────────────────────────────── */
/* Main card                                                   */
/* ─────────────────────────────────────────────────────────── */
function SearchInformationCard({ status, args, result }: {
  status: string; args: any; result: any;
}) {
  const queries: string[] = args?.queries ?? [];
  // CopilotKit 1.8.x uses "inProgress" during arg-streaming and "executing" during handler run
  const isLoading = status === "inProgress" || status === "executing";
  const isError = !!result?.error;

  /* Reset SSE store on first render of a new search to prevent stale data flash */
  const prevStatusRef = useRef(status);
  if (status === "inProgress" && prevStatusRef.current !== "inProgress") {
    resetSseStore();
  }
  prevStatusRef.current = status;

  /* ── Snapshot: freeze SSE data per-card when it transitions to complete ── */
  const snapshotRef = useRef<{ rounds: Record<string, any[]>; results: any[] } | null>(null);
  if (isLoading) {
    // While loading, keep snapshot clear so we read live SSE data
    snapshotRef.current = null;
  } else if (!snapshotRef.current) {
    // First render after complete → deep-copy current SSE data before next search wipes it
    snapshotRef.current = {
      rounds: { ..._sseStore.liveRounds },
      results: _sseStore.liveResults.map(r => r ? { ...r } : r),
    };
  }

  const [, setTick] = useState(0);
  useEffect(() => {
    const onTick = () => setTick(t => t + 1);
    window.addEventListener("ura-render-tick", onTick);
    // Polling fallback: re-read _sseStore every 300ms while loading
    const pollId = setInterval(() => {
      if (snapshotRef.current) return; // already frozen — skip
      const hasData =
        Object.keys(_sseStore.liveRounds).length > 0 ||
        _sseStore.liveResults.some(Boolean);
      if (hasData) setTick(t => t + 1);
    }, 300);
    return () => {
      window.removeEventListener("ura-render-tick", onTick);
      clearInterval(pollId);
    };
  }, [])

  // Use snapshot when complete, live SSE store when loading
  const liveRounds  = snapshotRef.current?.rounds  ?? _sseStore.liveRounds;
  const liveResults = snapshotRef.current?.results ?? _sseStore.liveResults;

  const finalResults: any[] = liveResults.filter(Boolean).length > 0
    ? liveResults.filter(Boolean)
    : (result?.query_results ?? []);

  const totalFAQs = finalResults.reduce((s, r) => s + (r.faq_results?.length ?? 0), 0);
  const totalDocs  = finalResults.reduce((s, r) => s + (r.doc_results?.length ?? 0), 0);

  // Count sub-queries whose retrieval is truly done (liveResults[i] populated)
  const doneCount   = queries.filter((_, i) => liveResults[i] != null).length;
  const allSubsDone = queries.length > 0 && doneCount === queries.length;
  const hasLiveData = Object.keys(liveRounds).length > 0 || liveResults.filter(Boolean).length > 0;

  const phase: Phase = isError
    ? "error"
    : !isLoading
    ? "complete"
    : allSubsDone
    ? "synthesizing"
    : hasLiveData
    ? "retrieving"
    : "starting";

  const elapsed = useElapsed(isLoading);

  /* Header config per phase */
  type Cfg = { subtitle: string; badgeColor: string };
  const cfg: Cfg = (() => {
    const n = queries.length;
    switch (phase) {
      case "starting":
        return { subtitle: "Analyzing your question and preparing search queries…", badgeColor: "#6366f1" };
      case "retrieving":
        return { subtitle: `Searching across FAQ and document sources — ${doneCount}/${n} completed`, badgeColor: "#6366f1" };
      case "synthesizing":
        return { subtitle: "All sources retrieved, preparing answer…", badgeColor: "#f59e0b" };
      case "complete":
        return { subtitle: `Found results from ${n} ${n > 1 ? "queries" : "query"} across FAQ and document sources`, badgeColor: "#22c55e" };
      case "error":
      default:
        return { subtitle: "Search encountered an error", badgeColor: "#ef4444" };
    }
  })();

  return (
    <div style={outerCard}>
      {/* ── Header ── */}
      <div style={{ display: "flex", alignItems: "flex-start", gap: 10, marginBottom: 12 }}>
        <div style={{ flex: 1 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ color: "#e2e8f0", fontWeight: 600, fontSize: 13 }}>Knowledge Base Search</span>
            {isLoading && (
              <span style={{ color: "#334155", fontSize: 11, fontVariantNumeric: "tabular-nums" }}>{elapsed}s</span>
            )}
            {phase === "complete" && elapsed > 0 && (
              <span style={{ color: "#334155", fontSize: 11 }}>{elapsed}s</span>
            )}
          </div>
          <div style={{ color: "#475569", fontSize: 11, marginTop: 2 }}>{cfg.subtitle}</div>
        </div>

        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" as const, justifyContent: "flex-end" }}>
          {queries.length > 0 && (
            <span style={pill("#64748b")}>{queries.length} sub-quer{queries.length === 1 ? "y" : "ies"}</span>
          )}
          {args?.attempt != null && args.attempt > 1 && (
            <span style={pill("#f59e0b")}>Attempt {args.attempt}/4</span>
          )}
        </div>
      </div>

      {/* ── Pipeline stepper + progress bar ── */}
      <PipelineStepper phase={phase} doneCount={doneCount} totalQueries={queries.length} />

      {/* ── Error ── */}
      {isError && (
        <div style={{ background: "#ef444411", border: "1px solid #ef444433", borderRadius: 8, padding: "10px 12px", color: "#f87171", fontSize: 13 }}>
          {result?.error ?? "Search failed."}
        </div>
      )}

      {/* ── Decomposed sub-queries list ── */}
      {queries.length > 0 && (
        <div style={{ marginBottom: 14 }}>
          <div style={sectionLabel("#6366f1")}>Decomposed Queries</div>
          <div style={{ display: "flex", flexDirection: "column" as const, gap: 5 }}>
            {queries.map((q: string, i: number) => {
              // ✓ checkmark only when liveResults[i] is truly populated
              const subDone    = liveResults[i] != null;
              const subRunning = isLoading && !subDone;
              return (
                <div key={i} style={{
                  background: "#1e2035",
                  border: `1px solid ${subDone ? "#22c55e33" : subRunning ? "#6366f133" : "#2d3148"}`,
                  borderRadius: 7, padding: "6px 10px",
                  display: "flex", alignItems: "center", gap: 8,
                }}>
                  <span style={{
                    width: 18, height: 18, borderRadius: "50%",
                    background: subDone ? "#22c55e22" : "#6366f122",
                    border: `1px solid ${subDone ? "#22c55e55" : "#6366f144"}`,
                    display: "inline-flex", alignItems: "center", justifyContent: "center",
                    fontSize: 9, fontWeight: 700, flexShrink: 0,
                    color: subDone ? "#22c55e" : "#818cf8",
                  }}>{i + 1}</span>
                  <span style={{ color: "#94a3b8", fontSize: 12 }}>{q}</span>
                  <span style={{ marginLeft: "auto" }}>
                    {subDone
                      ? <span style={{ ...pill("#22c55e"), fontSize: 9, padding: "1px 6px" }}>✓</span>
                      : subRunning
                      ? <Spinner size={10} color="#6366f1" />
                      : null}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ── Agent lanes ── */}
      {!isError && (
        <div>
          {/* Very early skeleton — before queries known */}
          {phase === "starting" && queries.length === 0 && (
            <div style={{ display: "flex", gap: 10, padding: "4px 0 8px" }}>
              <div style={{ flex: 1 }}>
                <div style={{ ...sectionLabel("#a78bfa"), display: "flex", gap: 5 }}>FAQ Search Agent <Spinner size={10} color="#a78bfa" /></div>
                {[90, 70, 50].map((w, i) => <Shimmer key={i} width={`${w}%`} color="#1a1d2e" highlight="#2d3158" delay={i * 0.2} />)}
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ ...sectionLabel("#38bdf8"), display: "flex", gap: 5 }}>Document Search Agent <Spinner size={10} color="#38bdf8" /></div>
                {[70, 55, 38].map((w, i) => <Shimmer key={i} width={`${w}%`} color="#0f1a2e" highlight="#1a3050" delay={i * 0.2} />)}
              </div>
            </div>
          )}

          {/* Per-query lanes — live or done */}
          {queries.length > 0 && (
            <>
              <div style={{ ...sectionLabel("#475569"), marginBottom: 8 }}>
                {phase === "complete"
                  ? "Search Results – FAQ Search Agent + Document Search Agent"
                  : "Searching…"}
              </div>

              {queries.map((q: string, i: number) => {
                const done = liveResults[i];
                if (done) {
                  return <DoneQueryLane key={i} idx={i} queryResult={done} />;
                }
                if (isLoading) {
                  return (
                    <LiveQueryLane key={i} idx={i} query={q}
                      faqRounds={liveRounds[`${i}-faq`] ?? []}
                      docRounds={liveRounds[`${i}-doc`] ?? []}
                    />
                  );
                }
                // status=complete but SSE missed — fallback to slim result
                const slim = result?.query_results?.[i];
                if (slim) return <DoneQueryLane key={i} idx={i} queryResult={slim} />;
                return (
                  <div key={i} style={{
                    background: "#13162a", border: "1px solid #2a2d44",
                    borderRadius: 10, padding: "10px 14px", marginBottom: 10,
                    display: "flex", alignItems: "center", gap: 8,
                  }}>
                    <div style={{
                      width: 22, height: 22, borderRadius: "50%",
                      background: "linear-gradient(135deg,#6366f1,#8b5cf6)",
                      display: "flex", alignItems: "center", justifyContent: "center",
                      fontSize: 10, fontWeight: 700, color: "#fff", flexShrink: 0,
                    }}>{i + 1}</div>
                    <span style={{ color: "#94a3b8", fontSize: 13 }}>{q}</span>
                    <span style={{ marginLeft: "auto", ...pill("#475569"), fontSize: 9 }}>No data</span>
                  </div>
                );
              })}
            </>
          )}

          {/* Synthesizing banner — smooth fade transition */}
          <div style={{
            display: "flex", alignItems: "center", gap: 10,
            background: "#1c1a0a", border: "1px solid #f59e0b33",
            borderRadius: 10, color: "#fcd34d", fontSize: 12,
            overflow: "hidden",
            maxHeight: phase === "synthesizing" ? 80 : 0,
            opacity: phase === "synthesizing" ? 1 : 0,
            padding: phase === "synthesizing" ? "10px 14px" : "0 14px",
            marginTop: phase === "synthesizing" ? 8 : 0,
            transition: "max-height 0.4s ease, opacity 0.3s ease, padding 0.3s ease, margin-top 0.3s ease",
          }}>
            <Spinner size={12} color="#fcd34d" />
            <div>
              <div style={{ fontWeight: 600, marginBottom: 2 }}>Preparing your answer</div>
              <div style={{ color: "#92400e", fontSize: 11 }}>
                Combining results from {queries.length} {queries.length !== 1 ? "queries" : "query"}…
              </div>
            </div>
          </div>

          {/* Summary stats — only shown when complete */}
          {phase === "complete" && finalResults.length > 0 && (() => {
            const hasAnswer = finalResults.some(r =>
              !isNegativeAnswer(r.faq_answer) || !isNegativeAnswer(r.doc_answer)
            );

            /* PATH signal — matches paper Section 3.4:
               A  = All sub-queries answered by at least one agent
               B  = Some sub-queries answered, others not
               C  = No agent answered, but retriever DID return items → query vague, clarify
               D  = Retriever returned 0 items → KB truly has nothing for this query */
            const allAnswered = finalResults.every(r =>
              !isNegativeAnswer(r.faq_answer) || !isNegativeAnswer(r.doc_answer)
            );
            const someAnswered = hasAnswer && !allAnswered;

            // PATH C vs D: did the retriever return ANY items at all?
            const hasRetrievedItems = totalFAQs > 0 || totalDocs > 0;

            const pathSignal = allAnswered
              ? { label: "PATH A", color: "#22c55e" }
              : someAnswered
              ? { label: "PATH B", color: "#22c55e" }
              : hasRetrievedItems
              ? { label: "PATH C", color: "#f59e0b" }
              : { label: "PATH D", color: "#ef4444" };

            return (
              <>
                {/* Stats bar */}
                <div style={{
                  display: "flex", alignItems: "center", gap: 10, marginTop: 8,
                  padding: "8px 12px", background: "#1a1d2e",
                  border: "1px solid #252840", borderRadius: 8,
                  flexWrap: "wrap" as const,
                }}>
                  <span style={{ color: "#475569", fontSize: 12 }}>Retrieved:</span>
                  <span style={pill("#a78bfa")}>{totalFAQs} FAQ entries</span>
                  <span style={pill("#38bdf8")}>{totalDocs} document passages</span>
                  {(totalFAQs > 0 && totalDocs > 0) && (
                    <span style={{ marginLeft: "auto", ...pill("#6366f1") }} title="Reciprocal Rank Fusion combines keyword (BM25) and semantic search results">RRF Hybrid Fusion</span>
                  )}
                </div>

                {/* PATH signal — separate prominent bar */}
                <div style={{
                  display: "flex", alignItems: "center", gap: 10, marginTop: 6,
                  padding: "8px 12px",
                  background: pathSignal.color + "0a",
                  border: `1px solid ${pathSignal.color}33`,
                  borderRadius: 8,
                }}>
                  <span style={{
                    width: 8, height: 8, borderRadius: "50%",
                    background: pathSignal.color, flexShrink: 0,
                    boxShadow: `0 0 6px ${pathSignal.color}66`,
                  }} />
                  <span style={{ color: pathSignal.color, fontSize: 12, fontWeight: 700 }}>
                    {pathSignal.label}
                  </span>
                  <span style={{ color: pathSignal.color + "cc", fontSize: 11 }}>
                    {pathSignal.label === "PATH A"
                      ? "Evidence found — answering directly"
                      : pathSignal.label === "PATH B"
                      ? "Partial evidence found — answering with available info"
                      : pathSignal.label === "PATH C"
                      ? "Related content exists but query too vague — expecting clarification"
                      : "No relevant information found in knowledge base"}
                  </span>
                </div>
              </>
            );
          })()}
        </div>
      )}

      {phase === "complete" && !isError && finalResults.length === 0 && queries.length > 0 && (
        <div style={{ color: "#64748b", fontSize: 13, marginTop: 4 }}>No results found across all sub-queries.</div>
      )}
    </div>
  );
}

/* ─────────────────────────────────────────────────────────── */
/* Hook registration                                           */
/* ─────────────────────────────────────────────────────────── */
export function ToolActions() {
  useCopilotAction({
    name: "search_information",
    description:
      "REQUIRED: Always call this before answering ANY question. Searches the knowledge base using parallel FAQ and document retrieval.\n\n" +
      "**DECOMPOSITION RULES (CRITICAL):**\n" +
      "- Each sub-query must be a SHORT KEYWORD PHRASE (2-5 words), NOT a full question.\n" +
      "- SINGLE-HOP (one entity, one fact): Put the keyword phrase in query1 only.\n" +
      "  Example: 'Kepler sinh năm nào?' → query1='Kepler năm sinh'\n" +
      "- MULTI-HOP (comparing/relating 2+ entities): Split into separate keyword phrases, one per entity.\n" +
      "  Example: 'Were Scott Derrickson and Ed Wood of the same nationality?'\n" +
      "  → query1='Scott Derrickson nationality'\n" +
      "  → query2='Ed Wood nationality'\n" +
      "  Example: 'Ai lớn tuổi hơn, Kepler hay Galileo?'\n" +
      "  → query1='Kepler năm sinh'\n" +
      "  → query2='Galileo năm sinh'\n" +
      "- COMPLEX (3+ entities/facts): Use all three query slots.\n" +
      "- All queries MUST be in the user's language.",
    parameters: [
      { name: "query1", type: "string", description: "First sub-query (required). For simple questions, put the full question here.", required: true },
      { name: "query2", type: "string", description: "Second sub-query for multi-hop questions (optional).", required: false },
      { name: "query3", type: "string", description: "Third sub-query for complex multi-hop questions (optional).", required: false },
      { name: "attempt", type: "number", description: "Current search attempt number. Pass 1 on first call, 2 on first retry, etc.", required: false },
    ],
    handler: async ({ query1, query2, query3 }: { query1: string; query2?: string; query3?: string }) => {
      const queries = [query1, query2, query3].filter((q): q is string => !!q && q.trim().length > 0);
      resetSseStore();
      try {
        const res = await fetch(`/api/search/stream`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ queries, top_k: 5 }),
        });
        if (!res.ok) {
          return { error: `HTTP ${res.status}: ${res.statusText}`, query_results: [] };
        }

        const { events } = await res.json();
        const slimResults: any[] = [];

        for (const event of events) {
          if (event.type === "query_done") {
            slimResults[event.query_idx] = {
              query: event.query,
              faq_answer: event.faq_answer,
              doc_answer: event.doc_answer,
              faq_results: event.faq_results ?? [],
              doc_results: event.doc_results ?? [],
              faq_attempts: event.faq_attempts,
              doc_attempts: event.doc_attempts,
              faq_steps: event.faq_steps ?? [],
              doc_steps: event.doc_steps ?? [],
            };
          }
          // Dispatch to module-level listener for UI updates
          try {
            if (typeof window !== "undefined") {
              window.dispatchEvent(new CustomEvent("ura-search-event", { detail: event }));
            }
          } catch { /* ignore */ }
          // Micro-delay between round events so React can render intermediate states
          if (event.type === "round") {
            await new Promise(r => setTimeout(r, 60));
          }
        }

        const llmResults = slimResults.filter(Boolean).map((r: any) => ({
          query: r.query,
          faq_answer: r.faq_answer,
          doc_answer: r.doc_answer,
          num_faq_results: r.faq_results?.length ?? 0,
          num_doc_results: r.doc_results?.length ?? 0,
        }));
        return {
          total_queries: queries.length,
          query_results: llmResults,
        };
      } catch (err: any) {
        return { error: err?.message ?? "Search request failed", query_results: [] };
      }
    },
    render: ({ status, args, result }: any) => {
      const queries = [args?.query1, args?.query2, args?.query3]
        .filter((q: any): q is string => !!q && q.trim().length > 0);
      return <SearchInformationCard status={status} args={{ ...args, queries }} result={result} />;
    },
  });

  return null;
}
