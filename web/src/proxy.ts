import { createHash, timingSafeEqual } from "node:crypto";
import { NextRequest, NextResponse } from "next/server";

export function proxy(request: NextRequest) {
  const username = process.env.DASHBOARD_USERNAME;
  const password = process.env.DASHBOARD_PASSWORD;
  const production =
    process.env.APP_ENV === "production" || process.env.VERCEL === "1";
  if (
    (production && (!username || !password || password.length < 16)) ||
    !!username !== !!password ||
    (process.env.APP_ENV &&
      !["local", "production"].includes(process.env.APP_ENV))
  ) {
    return new NextResponse("Dashboard access is not configured.", {
      status: 503,
    });
  }
  if (!username || !password) return NextResponse.next();
  const header = request.headers.get("authorization") || "";
  const supplied = header.startsWith("Basic ")
    ? Buffer.from(header.slice(6), "base64").toString("utf8")
    : "";
  const digest = (value: string) => createHash("sha256").update(value).digest();
  if (!timingSafeEqual(digest(supplied), digest(`${username}:${password}`))) {
    return new NextResponse("Authentication required.", {
      status: 401,
      headers: {
        "WWW-Authenticate": 'Basic realm="DecisionGuard", charset="UTF-8"',
        "Cache-Control": "no-store",
      },
    });
  }
  return NextResponse.next();
}
export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
