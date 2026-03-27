import {
  CopilotRuntime,
  LangChainAdapter,
  copilotRuntimeNextJSAppRouterEndpoint,
} from "@copilotkit/runtime";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { NextRequest } from "next/server";

// Allow long-running sub-agent retrieval (up to 3 LLM rounds × 2 agents)
export const maxDuration = 300;

const llm = new ChatGoogleGenerativeAI({
  model: "gemini-2.5-flash",
  apiKey: process.env.GOOGLE_API_KEY,
  temperature: 0.2,
  topP: 0.1,
  maxOutputTokens: 2048,
});

const serviceAdapter = new LangChainAdapter({
  chainFn: async ({ messages, tools }) => {
    // ── 1. Find last human message → start of current question ────────────
    const lastHumanIdx = messages.reduce(
      (idx: number, m: any, i: number) =>
        (m._getType?.() === "human" || m.role === "user") ? i : idx,
      0
    );

    // ── 2. Split cleanly: system | prior turns | current turn ─────────────
    const systemMsgs = messages.filter(
      (m: any) => m._getType?.() === "system" || m.role === "system"
    );
    const priorMsgs = messages.slice(0, lastHumanIdx).filter(
      (m: any) => m._getType?.() !== "system" && m.role !== "system"
    );
    // current turn = user msg + any tool_call/tool_result pairs so far
    const currentTurn = messages.slice(lastHumanIdx);

    // ── 3. Count searches done FOR THIS QUESTION ONLY ─────────────────────
    // Each completed search adds exactly 1 tool role message to currentTurn
    const searchesDone = currentTurn.filter(
      (m: any) => m._getType?.() === "tool" || m.role === "tool"
    ).length;

    // ── 4. Trim prior turns to cap context size ────────────────────────────
    // ~4 msgs per turn (human + ai_tool_call + tool_result + ai_answer)
    // Keep last 2 prior turns = last 8 messages before current question
    const trimmedPrior = priorMsgs.slice(-8);
    const activeMessages = [...systemMsgs, ...trimmedPrior, ...currentTurn];

    // ── DEBUG: Log message structure ──────────────────────────────────────
    console.log(`[CopilotKit chainFn] total=${messages.length} active=${activeMessages.length} searchesDone=${searchesDone} currentTurn=${currentTurn.length} prior=${trimmedPrior.length}`);
    for (const m of activeMessages) {
      const type = m._getType?.() ?? (m as any).role ?? "unknown";
      const content = typeof m.content === "string" ? m.content.slice(0, 80) : JSON.stringify(m.content).slice(0, 80);
      const extra = (m as any).tool_calls ? ` tool_calls=${JSON.stringify((m as any).tool_calls.map((tc: any) => tc.name || tc.function?.name))}` : "";
      const extra2 = (m as any).tool_call_id ? ` tool_call_id=${(m as any).tool_call_id}` : "";
      const extra3 = (m as any).name ? ` name=${(m as any).name}` : "";
      console.log(`  [${type}]${extra}${extra2}${extra3} ${content}...`);
    }

    // ── 5. Detect meta-question (never search for "what can you do?" etc.) ─
    const firstMsg = currentTurn[0];
    const lastText = (
      typeof firstMsg?.content === "string"
        ? firstMsg.content
        : Array.isArray(firstMsg?.content)
          ? firstMsg.content.map((c: any) => c?.text ?? "").join(" ")
          : ""
    ).toLowerCase();
    const isMetaQuestion =
      /(what (can|do) you|what topics|what (do you )?know|capabilities)/i.test(lastText);

    // ── 6. Tool binding strategy ───────────────────────────────────────────
    // Paper Algorithm 1: Manager decomposes → sub-agents retrieve → evaluate.
    // The "while t < T" retry loop in Algorithm 1 maps to the SUB-AGENTS'
    // internal reformulation (3 rounds each), NOT to outer Manager retries.
    // Manager calls search_information exactly ONCE per question; sub-agents
    // already do Refine(q,E) internally. Outer retries just waste 30s+ per
    // round with no new evidence (same KB, same decomposition).
    //
    //   searchesDone=0 → force search  (tool_choice: "any")
    //   searchesDone≥1 → must synthesize (PATH A/B/C→D — no more searches)
    if (isMetaQuestion || searchesDone >= 1) {
      console.log(`[CopilotKit chainFn] → Synthesis path (no tools). Starting llm.stream...`);
      try {
        const stream = await llm.stream(activeMessages);
        console.log(`[CopilotKit chainFn] → llm.stream() returned successfully`);
        return stream;
      } catch (err: any) {
        console.error(`[CopilotKit chainFn] → llm.stream() ERROR:`, err?.message ?? err);
        throw err;
      }
    }
    console.log(`[CopilotKit chainFn] → Tool-call path (tool_choice: "any"). Starting llm.bindTools...`);
    try {
      const stream = await llm
        .bindTools(tools, { tool_choice: "any" })
        .stream(activeMessages);
      console.log(`[CopilotKit chainFn] → bindTools.stream() returned successfully`);
      return stream;
    } catch (err: any) {
      console.error(`[CopilotKit chainFn] → bindTools.stream() ERROR:`, err?.message ?? err);
      throw err;
    }
  },
});

const runtime = new CopilotRuntime({});

export const POST = async (req: NextRequest) => {
  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    runtime,
    serviceAdapter,
    endpoint: "/api/copilotkit",
  });
  const res = await handleRequest(req);
  // Force browser to close the TCP connection after each response instead of
  // keeping it alive in the HTTP/1.1 pool.  Without this, streaming SSE
  // responses accumulate keep-alive connections (2 per question) and exhaust
  // the browser's 6-connection-per-hostname limit after ~3-4 questions.
  const headers = new Headers(res.headers);
  headers.set("Connection", "close");
  return new Response(res.body, {
    status: res.status,
    statusText: res.statusText,
    headers,
  });
};
