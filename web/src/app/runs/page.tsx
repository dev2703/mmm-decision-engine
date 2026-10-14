"use client";
import { useState } from "react";
import Link from "next/link";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { useWorkspace } from "@/components/workspace";
import {
  Badge,
  Empty,
  ErrorNotice,
  EvidenceDetails,
  Loading,
  PageTitle,
  Panel,
} from "@/components/ui";
import { date, useResource } from "@/lib/api";
import type { Optimization, Run, RunPage } from "@/lib/types";
import { OptimizationResult } from "@/components/optimization-result";

export default function Runs() {
  const { project } = useWorkspace();
  return <RunHistory key={project?.id} />;
}
function RunHistory() {
  const { project, revision, selectModel } = useWorkspace();
  const [kind, setKind] = useState("");
  const [status, setStatus] = useState("");
  const [offset, setOffset] = useState(0);
  const [selected, setSelected] = useState<Run>();
  const query = new URLSearchParams({
    limit: "20",
    offset: String(offset),
    ...(kind ? { kind } : {}),
    ...(status ? { status } : {}),
  });
  const runs = useResource<RunPage>(
    project ? `/projects/${project.id}/runs?${query}` : null,
    revision,
  );
  const active =
    selected &&
    runs.data?.items.find(
      (item) => item.id === selected.id && item.kind === selected.kind,
    );
  const optimization = useResource<Optimization>(
    active?.kind === "optimization" ? `/optimization-runs/${active.id}` : null,
    revision,
  );
  return (
    <>
      <PageTitle
        eyebrow="06 / RUNS & HISTORY"
        title="Every result has a history."
        description="Trace the dataset, configuration and evidence behind each training, evaluation and optimization job."
      />
      <Panel
        title="Run history"
        subtitle="Job completion and scientific release are separate states."
        action={
          <div className="filter-group">
            <select
              aria-label="Run type"
              value={kind}
              onChange={(event) => {
                setKind(event.target.value);
                setOffset(0);
              }}
            >
              <option value="">All types</option>
              {["training", "evaluation", "optimization"].map((value) => (
                <option key={value}>{value}</option>
              ))}
            </select>
            <select
              aria-label="Run status"
              value={status}
              onChange={(event) => {
                setStatus(event.target.value);
                setOffset(0);
              }}
            >
              <option value="">All statuses</option>
              {["QUEUED", "RUNNING", "SUCCEEDED", "FAILED"].map((value) => (
                <option key={value}>{value}</option>
              ))}
            </select>
          </div>
        }
      >
        <ErrorNotice error={runs.error} />
        {runs.loading ? (
          <Loading />
        ) : runs.data?.items.length ? (
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Run ID</th>
                  <th>Type</th>
                  <th>Status</th>
                  <th>Started</th>
                  <th>Model</th>
                  <th>Inspect</th>
                </tr>
              </thead>
              <tbody>
                {runs.data.items.map((run) => (
                  <tr key={`${run.kind}-${run.id}`}>
                    <td className="mono">{run.id.slice(0, 8)}</td>
                    <td className="capitalize">{run.kind}</td>
                    <td>
                      <Badge value={run.status} />
                    </td>
                    <td>{date(run.created_at)}</td>
                    <td>
                      <Link
                        href="/model-health"
                        className="text-link mono"
                        onClick={() => selectModel(run.model_run_id)}
                      >
                        {run.model_run_id.slice(0, 8)}
                      </Link>
                    </td>
                    <td>
                      <button
                        className="text-link"
                        onClick={() => setSelected(run)}
                      >
                        Evidence <ChevronRight size={13} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Empty description="No runs match the current project and filters." />
        )}
        <div className="pagination">
          <button
            className="button secondary"
            disabled={!offset}
            onClick={() => setOffset((value) => Math.max(0, value - 20))}
          >
            <ChevronLeft size={14} />
            Previous
          </button>
          <span>{offset ? `From run ${offset + 1}` : "Latest runs"}</span>
          <button
            className="button secondary"
            disabled={runs.data?.next_offset == null}
            onClick={() => setOffset(runs.data!.next_offset!)}
          >
            Next
            <ChevronRight size={14} />
          </button>
        </div>
      </Panel>
      {active && (
        <Panel
          title={`${active.kind} · ${active.id.slice(0, 8)}`}
          subtitle="Registered lineage and job metadata"
        >
          <EvidenceDetails title="Inspect run provenance" data={active} />
          {active.error_type && (
            <div className="notice danger">
              Job failed: {active.error_type}. No successful result is implied.
            </div>
          )}
        </Panel>
      )}
      <ErrorNotice error={optimization.error} />
      {optimization.data && <OptimizationResult result={optimization.data} />}
    </>
  );
}
