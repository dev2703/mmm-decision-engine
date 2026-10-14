import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
const id = "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}";
const reads = new RegExp(
  `^(capabilities|projects|projects/${id}(/(datasets|model-runs|model-comparison|runs))?|datasets/${id}/quality|model-runs/${id}(/(health|predictions|channels/[a-z0-9_]{1,80}))?|evaluation-runs/${id}/evidence|scenarios/${id}|optimization-runs/${id})$`,
);
const writes = new RegExp(
  `^(projects|projects/${id}/(datasets/generate|experiments)|experiments/${id}/run|model-runs/${id}/(evaluate|scenarios|constraints/validate)|scenarios/${id}/optimize|agent/query)$`,
);
const jsonError = (detail: string, status: number) =>
  NextResponse.json(
    { detail },
    { status, headers: { "Cache-Control": "no-store" } },
  );

async function proxy(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> },
) {
  const { path } = await context.params;
  const route = path.join("/");
  if (
    !(request.method === "GET" ? reads : writes).test(route) ||
    path.some((part) => !/^[a-z0-9_-]+$/.test(part))
  )
    return jsonError("Route unavailable", 404);
  const production =
    process.env.APP_ENV === "production" || process.env.VERCEL === "1";
  if (request.nextUrl.search.length > 2048)
    return jsonError("Query is too long", 414);
  if (request.method === "POST") {
    // Next's internal URL can use localhost behind a reverse proxy. Compare the
    // browser Origin with the actual Host; never trust forwarded-host overrides.
    let origin: URL;
    try {
      origin = new URL(request.headers.get("origin") || "");
    } catch {
      return jsonError("Same-origin request required", 403);
    }
    if (
      !["http:", "https:"].includes(origin.protocol) ||
      (production && origin.protocol !== "https:") ||
      origin.host !== request.headers.get("host") ||
      origin.origin !== request.headers.get("origin")
    )
      return jsonError("Same-origin request required", 403);
  }
  let base: URL;
  try {
    base = new URL(process.env.BACKEND_URL || "http://127.0.0.1:8000");
  } catch {
    return jsonError("Backend configuration is incomplete", 503);
  }
  if (
    !["http:", "https:"].includes(base.protocol) ||
    base.username ||
    base.password ||
    base.pathname !== "/" ||
    base.search ||
    base.hash ||
    (production &&
      (base.protocol !== "https:" || (process.env.API_TOKEN?.length || 0) < 32))
  )
    return jsonError("Backend configuration is incomplete", 503);
  let body: string | undefined;
  if (request.method === "POST") {
    if (
      request.headers.get("content-type")?.split(";")[0] !== "application/json"
    )
      return jsonError("JSON content required", 415);
    const reader = request.body?.getReader();
    const chunks: Uint8Array[] = [];
    let size = 0;
    if (reader) {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        size += value.byteLength;
        if (size > 32768) {
          await reader.cancel();
          return jsonError("Request is too large", 413);
        }
        chunks.push(value);
      }
    }
    body = Buffer.concat(chunks).toString("utf8");
  }
  try {
    const response = await fetch(
      new URL(`/${route}${request.nextUrl.search}`, base),
      {
        method: request.method,
        body,
        cache: "no-store",
        redirect: "error",
        signal: AbortSignal.timeout(30000),
        headers: {
          "Content-Type": "application/json",
          ...(process.env.API_TOKEN
            ? { Authorization: `Bearer ${process.env.API_TOKEN}` }
            : {}),
        },
      },
    );
    if (!response.headers.get("content-type")?.includes("application/json"))
      return jsonError("Backend returned an unexpected response", 502);
    return new NextResponse(await response.text(), {
      status: response.status,
      headers: {
        "Content-Type": "application/json",
        "Cache-Control": "no-store",
        ...(response.headers.get("x-correlation-id")
          ? { "X-Correlation-ID": response.headers.get("x-correlation-id")! }
          : {}),
      },
    });
  } catch {
    return jsonError(
      "Backend is unavailable or the request timed out. Refresh to check queued work before retrying a write.",
      502,
    );
  }
}
export { proxy as GET, proxy as POST };
