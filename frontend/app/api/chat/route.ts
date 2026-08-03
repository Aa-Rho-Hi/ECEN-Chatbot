/**
 * API route: POST /api/chat
 * Proxies the request to the FastAPI backend and streams the response back.
 * This avoids CORS issues and lets the frontend talk to /api/chat directly.
 */

import { NextRequest } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8000";

/**
 * Render a user-facing message as a valid SSE stream.
 *
 * The client reads this endpoint with a streaming reader and only understands
 * `data:` frames, so an error returned as plain text or JSON is invisible to it
 * — it just throws and shows a generic apology. Wrapping the message in SSE
 * means an outage explains itself in the chat transcript.
 */
function sseError(message: string, status: number): Response {
  const body =
    `event: sources\ndata: []\n\n` +
    `data: ${message.replace(/\n/g, "\\n")}\n\n` +
    `data: [DONE]\n\n`;
  return new Response(body, {
    status,
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
    },
  });
}

// A single attempt can't hang forever: the backend streams SSE and can sit idle
// between tokens, but if it never responds at all we must give up rather than
// hold the user's request open until Cloud Run's 300s request timeout.
const ATTEMPT_TIMEOUT_MS = 120_000;
// Total wall-clock budget for connecting. Cold starts need a long window (pull a
// multi-GB image, then load two ML models before the port opens), but the old
// 90-attempt loop meant a genuinely crashed backend kept users on a spinner for
// 4.5 minutes before showing a generic error. Bounding by elapsed time keeps
// the cold-start case working while failing a hard-down backend sooner.
const CONNECT_BUDGET_MS = 150_000;
const RETRY_DELAY_MS = 3_000;

/**
 * The FastAPI backend loads ML models on startup (~60-90s after a cold start
 * or deploy). Until it binds its port, fetch fails with ECONNREFUSED. Retry
 * instead of surfacing an error to the user.
 */
async function fetchBackendWithRetry(body: unknown, clientIp: string): Promise<Response> {
  const deadline = Date.now() + CONNECT_BUDGET_MS;
  let lastErr: unknown;
  for (let attempt = 1; ; attempt++) {
    try {
      return await fetch(`${BACKEND_URL}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          // Forward the real client IP (set by Cloud Run on the outer request)
          // so the backend's per-IP rate limiting sees users, not localhost.
          "x-forwarded-for": clientIp,
        },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(ATTEMPT_TIMEOUT_MS),
      });
    } catch (err) {
      lastErr = err;
      if (Date.now() + RETRY_DELAY_MS >= deadline) {
        console.error(`[api/chat] backend unreachable after ${attempt} attempts`, err);
        throw err;
      }
      await new Promise(r => setTimeout(r, RETRY_DELAY_MS));
    }
  }
  // Unreachable, but keeps the compiler honest about lastErr being used.
  throw lastErr;
}

export async function POST(req: NextRequest) {
  const body = await req.json();
  const clientIp = req.headers.get("x-forwarded-for")?.split(",")[0]?.trim() ?? "unknown";

  let upstream: Response;
  try {
    upstream = await fetchBackendWithRetry(body, clientIp);
  } catch {
    // Backend never answered. Return SSE rather than a bare error body: the
    // client reads this response as a stream, so a non-SSE payload surfaces as
    // the generic "something went wrong" instead of an honest explanation.
    return sseError(
      "Sorry — I can't reach my knowledge base right now. This is a temporary " +
      "problem on my end, not a problem with your question. Please try again in a moment.",
      503,
    );
  }

  if (!upstream.ok) {
    // Never forward the upstream body verbatim: FastAPI's default 500 body is
    // the string "Internal Server Error", which is what users saw during the
    // 2026-07-26 database outage. Rate limiting keeps its own status so the UI
    // can show its specific message.
    const detail = await upstream.text().catch(() => "");
    console.error(`[api/chat] backend ${upstream.status}: ${detail.slice(0, 300)}`);
    if (upstream.status === 429) {
      return new Response(detail, { status: 429 });
    }
    return sseError(
      "Sorry — something went wrong on my end while answering that. " +
      "Please try again in a moment.",
      upstream.status >= 500 ? 503 : upstream.status,
    );
  }

  // Stream the SSE response straight through
  return new Response(upstream.body, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}
