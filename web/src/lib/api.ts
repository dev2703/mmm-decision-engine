"use client";

import { useEffect, useState } from "react";

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public requestId: string | null,
  ) {
    super(message);
  }
}
export async function api<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const response = await fetch(`/api/backend${path}`, {
    ...options,
    cache: "no-store",
    headers: { "Content-Type": "application/json", ...options.headers },
  });
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const detail =
      typeof data?.detail === "string"
        ? data.detail
        : response.status === 422
          ? "Check the values and constraints, then try again."
          : "Evidence could not be retrieved. Please try again.";
    throw new ApiError(
      detail,
      response.status,
      response.headers.get("X-Correlation-ID"),
    );
  }
  return data as T;
}
export function useResource<T>(path: string | null, revision = 0) {
  const key = `${path}:${revision}`;
  const [state, setState] = useState<{ key: string; data?: T; error?: Error }>({
    key: "",
  });
  useEffect(() => {
    if (!path) return;
    const controller = new AbortController();
    api<T>(path, { signal: controller.signal })
      .then((data) => {
        if (!controller.signal.aborted) setState({ key, data });
      })
      .catch((error) => {
        if (!controller.signal.aborted) setState({ key, error });
      });
    return () => controller.abort();
  }, [path, key]);
  return {
    data: state.key === key ? state.data : undefined,
    error: state.key === key ? state.error : undefined,
    loading: !!path && (state.key !== key || (!state.data && !state.error)),
  };
}
export const money = (value: number | null | undefined, compact = false) =>
  value == null || !Number.isFinite(value)
    ? "—"
    : new Intl.NumberFormat("en-AU", {
        style: "currency",
        currency: "AUD",
        maximumFractionDigits: 0,
        notation: compact ? "compact" : "standard",
      }).format(value);
export const number = (value: number | null | undefined, digits = 2) =>
  value == null || !Number.isFinite(value)
    ? "—"
    : new Intl.NumberFormat("en-AU", { maximumFractionDigits: digits }).format(
        value,
      );
export const percent = (value: number | null | undefined) =>
  value == null
    ? "—"
    : new Intl.NumberFormat("en-AU", {
        style: "percent",
        maximumFractionDigits: 1,
      }).format(value);
export const channelName = (value: string) =>
  (
    ({
      search_spend: "Search",
      meta_spend: "Meta",
      tv_spend: "TV",
      ooh_spend: "Out of home",
      youtube_spend: "YouTube",
    }) as Record<string, string>
  )[value] || value.replaceAll("_", " ");
export const date = (value: string) =>
  new Date(
    value.length === 10 ? `${value}T12:00:00Z` : value,
  ).toLocaleDateString("en-AU", {
    day: "numeric",
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  });
