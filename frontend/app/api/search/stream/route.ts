import { NextRequest } from "next/server";

export const dynamic = "force-dynamic";
export const maxDuration = 300;

const BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8005";

/**
 * Buffered JSON proxy — fetches the backend SSE stream, collects all events,
 * and returns them as a single JSON response.  This avoids long-lived
 * streaming connections that exhaust the browser's HTTP/1.1 connection pool
 * (6 per hostname) after several questions in a session.
 */
export async function POST(req: NextRequest) {
  const body = await req.json();
  const upstream = await fetch(`${BACKEND_URL}/search/parallel/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!upstream.ok || !upstream.body) {
    return Response.json({ error: `Upstream ${upstream.status}` }, { status: 502 });
  }

  const reader = upstream.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  const events: unknown[] = [];

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop()!;
    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      try { events.push(JSON.parse(line.slice(6))); } catch { /* skip malformed */ }
    }
  }

  return Response.json({ events });
}
