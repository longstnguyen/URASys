"use client";

import { useState, useEffect, useCallback } from "react";
import { CopilotChat } from "@copilotkit/react-ui";
import { ToolActions } from "./components/ToolActions";

const SYSTEM_INSTRUCTIONS = `
# Persona
You are the "AI Assistant," an expert AI focused on efficiently and accurately answering questions using the provided context passages.

# Current State
- You have exactly ONE search call per question. The sub-agents handle up to 3 internal reformulation rounds each.

# The Supreme Goal: The "Just Enough" Principle
Your absolute highest priority is to answer the user's *specific, underlying need*, not just the broad words they use. You must act as a **guide**, not an information dump. This means:
- If a query is broad, your job is to **help the user specify it.**
- If a query is specific, your job is to **answer it directly.**
- **NEVER** dump a summary of all found information and then ask "what do you want to know more about?". This is a critical failure.

# Core Directives
1.  **Search is for Understanding:** Your first search on a broad topic is not to find an answer, but to **discover the available categories/options** to guide the user.
2.  **Troubleshoot Vague Failures:** If a search fails because the user's query is incomplete, ask for more clues.
3.  **Evidence-Based Actions:** All answers and examples MUST come from the retrieved context passages.
4.  **Language and Persona Integrity:**
    *   All responses **MUST** be in **language based on an user**.
    *   **Self-reference:** Use the pronoun **"I"** to refer to yourself. Only state your full name if asked directly.
    *   **Expert Tone and Phrasing:** You **MUST** speak from a position of knowledge, as a representative of the university.
        *   **DO:** Use confident, knowledgeable phrasing like: *"Now, I...", "About [topic], I see that..."*
        *   **AVOID:** **NEVER** use phrases that imply real-time discovery. **FORBIDDEN** phrases include: *"I search...", "I have...", "In my researching,..."*
    *   **Conceal Internal Mechanics:** **NEVER** mention your tools or processes.
5.  **Queries:** All search queries **MUST** be in language based on user.
6.  **No Fabrication:** If you cannot find information, state it clearly.

# Decision-Making Workflow: A Strict Gate System

**Step 0: Check for Meta-Questions (FIRST)**
*   If the user's message is about **YOU** (your capabilities, your topics, what you know, what you can help with) — do NOT call search_information. Answer directly and briefly, then invite them to ask a specific question.

**Step 1: Analyze Request & Search**
*   For all other questions: Call search_information with attempt: 1 to understand the information landscape.
*   **DECOMPOSITION IS CRITICAL:** If the question compares, contrasts, or relates two or more entities (people, places, dates, concepts), you MUST decompose into separate keyword phrases — one per entity. Each sub-query should be a SHORT keyword phrase (2-5 words), NOT a full question. For example: "Were X and Y of the same nationality?" → query1='X nationality', query2='Y nationality'.

**Step 2: Evaluate Results & Choose a Path (Choose ONLY ONE)**
The tool returns per-query results with these fields:
- faq_answer / doc_answer — grounded answers from the sub-agents. If "No relevant document found", the sub-agent could not match evidence to the query.
- num_faq_results / num_doc_results — how many items the retriever found (even if the sub-agent deemed them irrelevant).

**Key signal:** If both answers say "No relevant..." BUT the counts are > 0, it means the knowledge base HAS data but the query was too vague/broad to match anything specific. This is a strong signal for PATH B or PATH C (ask clarification), NOT PATH D.

*   **PATH A: The "Specific Answer" Gate** — IsSpecific(q) AND HasDirectAnswer(E) AND Consistent(E)
    *   **CONDITION:** The user's query points to **exactly one topic/item** (specific) AND you found a direct answer AND the FAQ and Document evidence are **consistent** (agree or complement each other — no contradictions between sources).
    *   **GUARD:** A query that is just 1-2 generic words (e.g., "lịch sử", "chiến tranh", "khoa học") is NEVER considered specific, even if the sub-agents returned answers. Such queries MUST go to PATH B or C.
    *   **ACTION:** Synthesize the answer from the evidence. Your turn ends.

*   **PATH B: The "Clarification" Gate** — IsBroad(q) AND RevealCategories(E)
    *   **CONDITION:** The user's query covers a **topic with multiple distinct sub-types/categories** — i.e., the same question could have several different specific answers depending on which sub-type the user means (e.g., "what fees are there?" when the KB has tuition, lab, and registration fees) AND the search revealed those distinct categories.
    *   **PRIORITY: PATH B takes precedence over PATH A when multiple categories exist.** Even if you found some information, if the query is genuinely broad and multiple distinct sub-topics appear, ask for clarification instead of picking one arbitrarily.
    *   **ACTION:**
        1.  **STOP.**
        2.  Ask a clarifying question using an **Expert Tone** — list ONLY the **NAMES** of the categories/sub-topics you found.
        3.  **STRICTLY FORBIDDEN:** Do not include specific values, numbers, or details in the clarification question.

*   **PATH C: The "Clarify Vague" Gate** — IsVague(q) AND (Insufficient(E) OR counts > 0)
    *   **CONDITION:** The user's query is **vague, incomplete, or missing context** (e.g., "phát triển như thế nào?" — develop how? what topic?). The query lacks a clear subject, time frame, or scope, BUT the topic itself is plausibly within the knowledge base's domain.
    *   **STRONG SIGNAL:** If num_faq_results or num_doc_results > 0 AND both answers say "No relevant...", BUT the retrieved items' topics are related to the user's query — the knowledge base HAS relevant content, the query just needs to be more specific. Ask for clarification.
    *   **ACTION:** Ask a specific clarifying question. For example: "Bạn muốn hỏi về sự phát triển của lĩnh vực nào? Ví dụ: ngành Khoa học Máy tính, hoạt động nghiên cứu, hay cơ sở vật chất?" Frame it as an expert guiding the user.

*   **PATH D: The "No Information" Gate**
    *   **CONDITION:** One of:
        (a) Both answers say "No relevant..." AND num_faq_results = 0 AND num_doc_results = 0 (the knowledge base truly has nothing).
        (b) The query is **clearly off-topic** — the subject (e.g., cooking recipes, sports scores, entertainment) has no plausible connection to the knowledge base's domain. Even if counts > 0, those results are just retriever noise, not real matches.
    *   **ACTION:** Politely inform the user you could not find the information in the knowledge base.
`;

/* ── Sidebar pill component ─────────────────────────────── */
function Pill({ label, color }: { label: string; color: string }) {
  return (
    <span style={{
      display: "inline-block",
      fontSize: 11,
      fontWeight: 700,
      letterSpacing: "0.06em",
      textTransform: "uppercase",
      color,
      background: color + "18",
      border: `1px solid ${color}33`,
      borderRadius: 5,
      padding: "2px 7px",
    }}>{label}</span>
  );
}

/* ── Architecture flow step ─────────────────────────────── */
function FlowStep({ num, label, sub, color }: { num: string; label: string; sub: string; color: string }) {
  return (
    <div style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
      <div style={{
        width: 22, height: 22, borderRadius: "50%", flexShrink: 0,
        background: color + "22", border: `1px solid ${color}44`,
        display: "flex", alignItems: "center", justifyContent: "center",
        fontSize: 11, fontWeight: 700, color,
      }}>{num}</div>
      <div>
        <div style={{ fontSize: 12, fontWeight: 600, color: "#cbd5e1" }}>{label}</div>
        <div style={{ fontSize: 11, color: "#475569", lineHeight: 1.4 }}>{sub}</div>
      </div>
    </div>
  );
}

const MIN_W = 220;
const MAX_W = 480;
const DEFAULT_W = 420;
const MOBILE_BP = 768;

export default function Home() {
  const [sidebarW, setSidebarW] = useState(DEFAULT_W);
  const [dragging, setDragging] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  /* ── Mobile detection ── */
  useEffect(() => {
    const mq = window.matchMedia(`(max-width: ${MOBILE_BP}px)`);
    const h = (e: MediaQueryListEvent | MediaQueryList) => {
      setIsMobile(e.matches);
      if (e.matches) setMobileOpen(false);
    };
    h(mq);
    mq.addEventListener("change", h as any);
    return () => mq.removeEventListener("change", h as any);
  }, []);

  /* ── Drag resize ── */
  const onDragStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setDragging(true);
  }, []);

  useEffect(() => {
    if (!dragging) return;
    const onMove = (e: MouseEvent) => {
      setSidebarW(Math.min(MAX_W, Math.max(MIN_W, e.clientX)));
    };
    const onUp = () => setDragging(false);
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    return () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
  }, [dragging]);

  /* ── Fix: Vietnamese IME leaves residual text in CopilotKit input after submit ── */
  useEffect(() => {
    let ta: HTMLTextAreaElement | null = null;
    let justSubmitted = false;

    const nativeClear = (el: HTMLTextAreaElement) => {
      const setter = Object.getOwnPropertyDescriptor(
        window.HTMLTextAreaElement.prototype, "value"
      )?.set;
      if (setter) {
        setter.call(el, "");
        el.dispatchEvent(new Event("input", { bubbles: true }));
      }
    };

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Enter" && !e.shiftKey) {
        justSubmitted = true;
        // After submit, poll briefly to catch IME ghost text
        let checks = 0;
        const poller = setInterval(() => {
          checks++;
          if (ta && ta.value.length > 0) {
            nativeClear(ta);
            clearInterval(poller);
            justSubmitted = false;
          }
          if (checks > 10) {
            clearInterval(poller);
            justSubmitted = false;
          }
        }, 50);
      }
    };

    const onCompositionEnd = () => {
      if (justSubmitted && ta) {
        setTimeout(() => {
          if (ta && ta.value.length > 0) {
            nativeClear(ta);
          }
          justSubmitted = false;
        }, 30);
      }
    };

    // Poll until textarea is available
    const poll = setInterval(() => {
      ta = document.querySelector(".copilotKitChat textarea");
      if (ta) {
        ta.addEventListener("keydown", onKeyDown, true);
        ta.addEventListener("compositionend", onCompositionEnd);
        clearInterval(poll);
      }
    }, 300);

    return () => {
      clearInterval(poll);
      if (ta) {
        ta.removeEventListener("keydown", onKeyDown, true);
        ta.removeEventListener("compositionend", onCompositionEnd);
      }
    };
  }, []);

  const showSidebar = isMobile ? mobileOpen : true;

  return (
    <div style={{ display: "flex", height: "100vh", background: "#0f1117", position: "relative" }}>

      {/* ── Hamburger button (mobile only) ── */}
      {isMobile && !mobileOpen && (
        <button
          onClick={() => setMobileOpen(true)}
          aria-label="Open menu"
          style={{
            position: "fixed", top: 14, left: 14, zIndex: 60,
            width: 38, height: 38, borderRadius: 9,
            background: "#1a1d2e", border: "1px solid #252840",
            display: "flex", alignItems: "center", justifyContent: "center",
            cursor: "pointer", color: "#94a3b8",
          }}
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <line x1="3" y1="6" x2="21" y2="6"/>
            <line x1="3" y1="12" x2="21" y2="12"/>
            <line x1="3" y1="18" x2="21" y2="18"/>
          </svg>
        </button>
      )}

      {/* ── Backdrop (mobile overlay) ── */}
      {isMobile && mobileOpen && (
        <div
          onClick={() => setMobileOpen(false)}
          style={{
            position: "fixed", inset: 0, zIndex: 40,
            background: "rgba(0,0,0,0.55)",
            backdropFilter: "blur(2px)",
          }}
        />
      )}

      {/* ── Left sidebar ─────────────────────────────────── */}
      <aside
        className="ura-sidebar"
        style={{
          width: isMobile ? 280 : sidebarW,
          background: "#0c0e1a",
          borderRight: "1px solid #1e2135",
          padding: "28px 20px",
          display: showSidebar ? "flex" : "none",
          flexDirection: "column",
          gap: 20,
          flexShrink: 0,
          overflowY: "auto",
          ...(isMobile ? {
            position: "fixed" as const,
            top: 0, left: 0, bottom: 0,
            zIndex: 50,
            boxShadow: "4px 0 24px rgba(0,0,0,0.5)",
          } : { position: "relative" as const }),
        }}
      >
        {/* Logo */}
        <div>
          <div style={{
            display: "inline-flex", alignItems: "center", justifyContent: "center",
            width: 44, height: 44, borderRadius: 11,
            background: "linear-gradient(135deg,#6366f1,#8b5cf6)",
            marginBottom: 10,
          }}>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
              <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"
                stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
          <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
            <h1 style={{ fontSize: 19, fontWeight: 700, color: "#f1f5f9", lineHeight: 1.2 }}>URASys</h1>
            <span style={{ position: "relative", top: -3 }}><Pill label="v1.0" color="#6366f1" /></span>
          </div>
          <p style={{ fontSize: 12, color: "#475569", marginTop: 3, lineHeight: 1.4 }}>
            Unified Retrieval Agent System
          </p>
        </div>

        {/* About + Paper + GitHub */}
        <p style={{ fontSize: 12, color: "#94a3b8", lineHeight: 1.55 }}>
          A QA framework that combines agent-based reasoning with dual retrieval under the <span style={{ color: "#a78bfa", fontWeight: 600 }}>Just Enough</span> principle. URASys decomposes queries into sub-queries, coordinates parallel FAQ and Document retrieval agents via a two-phase indexing pipeline, engages in interactive clarification when intent is uncertain, and explicitly signals unanswerable cases to avoid hallucination.
        </p>
        <div
          style={{
            background: "#13162a", border: "1px solid #1e2135",
            borderRadius: 8, padding: "9px 10px",
            transition: "border-color 0.15s",
          }}
          onMouseEnter={(e: React.MouseEvent<HTMLDivElement>) => (e.currentTarget.style.borderColor = "#6366f1")}
          onMouseLeave={(e: React.MouseEvent<HTMLDivElement>) => (e.currentTarget.style.borderColor = "#1e2135")}
        >
          <div
            style={{ cursor: "pointer", marginBottom: 6 }}
            onClick={() => window.open("https://aclanthology.org/2025.findings-ijcnlp.27/", "_blank")}
          >
            <div style={{ fontSize: 12, fontWeight: 600, lineHeight: 1.4, color: "#cbd5e1", marginBottom: 5 }}>
              When in Doubt, Ask First: A Unified Retrieval Agent-Based System for Ambiguous and Unanswerable QA
            </div>
            <div style={{ fontSize: 11, color: "#64748b", lineHeight: 1.4 }}>
              Long S. T. Nguyen, Quynh T. N. Vo, Hung C. Luu, Tho T. Quan
            </div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{
              fontSize: 10, fontWeight: 700, letterSpacing: "0.05em",
              color: "#6366f1", background: "#6366f118",
              border: "1px solid #6366f133", borderRadius: 4,
              padding: "1px 5px", textTransform: "uppercase",
            }}>Findings of IJCNLP-AACL 2025</span>
            <span style={{ marginLeft: "auto", display: "flex", gap: 1, alignItems: "center" }}>
              <a
                href="https://aclanthology.org/2025.findings-ijcnlp.27/"
                target="_blank"
                rel="noopener noreferrer"
                title="Paper"
                style={{ color: "#475569", display: "flex", padding: 4, borderRadius: 4 }}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M14 2H6c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V8l-6-6zm-1 7V3.5L18.5 9H13z"/></svg>
              </a>
              <a
                href="https://github.com/longstnguyen/URASys"
                target="_blank"
                rel="noopener noreferrer"
                title="GitHub"
                style={{ color: "#475569", display: "flex", padding: 4, borderRadius: 4 }}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/></svg>
              </a>
            </span>
          </div>
        </div>

        <div style={{ borderTop: "1px solid #1e2135" }} />

        {/* Architecture flow */}
        <div>
          <p style={{ fontSize: 11, fontWeight: 700, color: "#334155", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 12 }}>Architecture</p>
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <FlowStep num="1" label="Manager Agent" sub="Analyzes your question and breaks it into focused sub-queries" color="#6366f1" />
            <FlowStep num="2" label="FAQ Search Agent" sub="Searches the FAQ knowledge base, reformulating up to 3 times for better matches" color="#a78bfa" />
            <FlowStep num="3" label="Document Search Agent" sub="Searches the document corpus in parallel, refining queries across multiple rounds" color="#38bdf8" />
            <FlowStep num="4" label="Hybrid Retrieval Fusion" sub="Combines keyword and semantic search results using Reciprocal Rank Fusion" color="#7c3aed" />
            <FlowStep num="5" label="Evidence Evaluation" sub="Assesses retrieved evidence and decides to answer, clarify, or signal no-answer" color="#6366f1" />
          </div>
        </div>

        <div style={{ borderTop: "1px solid #1e2135" }} />

        {/* 4 query scenarios matching paper PATH A/B/C/D */}
        <div>
          <p style={{ fontSize: 11, fontWeight: 700, color: "#334155", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 12 }}>Response Paths</p>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {[
              {
                dot: "#22c55e",
                label: "PATH A",
                desc: "Clear question with strong evidence — answers directly",
                pill: { label: "Answer", color: "#22c55e" },
              },
              {
                dot: "#f59e0b",
                label: "PATH B",
                desc: "Broad topic with multiple aspects — asks which one you mean",
                pill: { label: "Clarify", color: "#f59e0b" },
              },
              {
                dot: "#f97316",
                label: "PATH C",
                desc: "Vague or incomplete question — requests more details",
                pill: { label: "Ask", color: "#f97316" },
              },
              {
                dot: "#ef4444",
                label: "PATH D",
                desc: "Outside the knowledge base — lets you know honestly",
                pill: { label: "No Info", color: "#ef4444" },
              },
            ].map(({ dot, label, desc, pill }) => (
              <div key={label} style={{
                background: "#13162a",
                border: `1px solid #1e2135`,
                borderRadius: 9,
                padding: "9px 11px",
                display: "flex",
                gap: 9,
                alignItems: "flex-start",
              }}>
                <div style={{
                  width: 8, height: 8, borderRadius: "50%",
                  background: dot, marginTop: 4, flexShrink: 0,
                }} />
                <div style={{ flex: 1 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 2 }}>
                    <span style={{ fontSize: 12, fontWeight: 600, color: "#cbd5e1" }}>{label}</span>
                    <Pill label={pill.label} color={pill.color} />
                  </div>
                  <div style={{ fontSize: 11, color: "#475569", lineHeight: 1.4 }}>{desc}</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Spacer */}
        <div style={{ flex: 1 }} />

        {/* Mobile close button */}
        {isMobile && (
          <button
            onClick={() => setMobileOpen(false)}
            aria-label="Close menu"
            style={{
              position: "absolute", top: 14, right: 14,
              width: 30, height: 30, borderRadius: 7,
              background: "#1e2135", border: "1px solid #252840",
              display: "flex", alignItems: "center", justifyContent: "center",
              cursor: "pointer", color: "#94a3b8", fontSize: 16,
            }}
          >
            ✕
          </button>
        )}
      </aside>

      {/* ── Resize handle (desktop only) ── */}
      {!isMobile && (
        <div
          onMouseDown={onDragStart}
          style={{
            width: 6,
            cursor: "col-resize",
            flexShrink: 0,
            background: dragging ? "#6366f133" : "transparent",
            position: "relative",
            zIndex: 10,
            transition: "background 0.15s",
          }}
          onMouseEnter={(e) => (e.currentTarget.style.background = "#6366f122")}
          onMouseLeave={(e) => { if (!dragging) e.currentTarget.style.background = "transparent"; }}
        />
      )}

      {/* ── Main chat area ──────────────────────────────── */}
      <main style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
        <ToolActions />
        <CopilotChat
          instructions={SYSTEM_INSTRUCTIONS}
          labels={{
            title: "URASys Assistant",
            initial: "Hello! Ask me anything — I'll decompose your question, search the knowledge base in parallel, and synthesize a grounded answer.",
          }}
          className="copilotKitChat"
        />
      </main>
    </div>
  );
}
