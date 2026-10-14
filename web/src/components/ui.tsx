"use client";
import {
  AlertCircle,
  ArrowUpRight,
  Check,
  Database,
  LoaderCircle,
} from "lucide-react";
import Link from "next/link";

export function Badge({ value }: { value: string }) {
  const tone = /BLOCK|FAILED|BLOCKER/.test(value)
    ? "danger"
    : /WARN|RESTRICT|RUNNING|QUEUED|CANDIDATE/.test(value)
      ? "warning"
      : /PASS|RESOLVED|SUCCEEDED/.test(value)
        ? "positive"
        : "neutral";
  return (
    <span className={`badge ${tone}`}>
      <span />
      {value.replaceAll("_", " ")}
    </span>
  );
}
export function PageTitle({
  eyebrow,
  title,
  description,
  children,
}: {
  eyebrow: string;
  title: string;
  description: string;
  children?: React.ReactNode;
}) {
  return (
    <header className="page-heading">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {children}
    </header>
  );
}
export function Panel({
  title,
  subtitle,
  children,
  action,
  className = "",
}: {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
  action?: React.ReactNode;
  className?: string;
}) {
  return (
    <section className={`panel ${className}`}>
      <header className="panel-heading">
        <div>
          <h2>{title}</h2>
          {subtitle && <p>{subtitle}</p>}
        </div>
        {action}
      </header>
      {children}
    </section>
  );
}
export function Empty({
  title = "No evidence yet",
  description,
  href,
  action,
}: {
  title?: string;
  description: string;
  href?: string;
  action?: string;
}) {
  return (
    <div className="empty">
      <Database size={25} strokeWidth={1.5} />
      <h3>{title}</h3>
      <p>{description}</p>
      {href && (
        <Link className="button secondary" href={href}>
          {action || "Explore workspace"}
          <ArrowUpRight size={15} />
        </Link>
      )}
    </div>
  );
}
export function Loading() {
  return (
    <div className="loading" role="status">
      <LoaderCircle className="spin" size={18} /> Loading verified evidence…
    </div>
  );
}
export function ErrorNotice({ error }: { error?: Error | null }) {
  return error ? (
    <div className="notice danger" role="alert">
      <AlertCircle size={19} />
      <div>
        <strong>Evidence unavailable</strong>
        <p>{error.message}</p>
      </div>
    </div>
  ) : null;
}
export function Notice({
  title,
  children,
  positive = false,
}: {
  title: string;
  children: React.ReactNode;
  positive?: boolean;
}) {
  return (
    <div className={`notice ${positive ? "positive" : "warning"}`}>
      {positive ? <Check size={20} /> : <AlertCircle size={20} />}
      <div>
        <strong>{title}</strong>
        <p>{children}</p>
      </div>
    </div>
  );
}
export function EvidenceDetails({
  data,
  title = "Inspect evidence",
}: {
  data: unknown;
  title?: string;
}) {
  return (
    <details className="evidence-details">
      <summary>{title}</summary>
      <pre>{JSON.stringify(data, null, 2)}</pre>
    </details>
  );
}
export function Stat({
  label,
  value,
  detail,
}: {
  label: string;
  value: React.ReactNode;
  detail: string;
}) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      <div className="stat-detail">{detail}</div>
    </div>
  );
}
