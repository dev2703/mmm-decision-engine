"use client";
import { useState } from "react";
import { useWorkspace } from "@/components/workspace";
import {
  Badge,
  Empty,
  ErrorNotice,
  EvidenceDetails,
  Loading,
  PageTitle,
  Panel,
  Stat,
} from "@/components/ui";
import { date, useResource } from "@/lib/api";
import type { Quality } from "@/lib/types";

export default function DataQuality() {
  const { dataset, revision } = useWorkspace();
  const quality = useResource<Quality>(
    dataset ? `/datasets/${dataset.id}/quality` : null,
    revision,
  );
  const [filter, setFilter] = useState("ALL");
  return (
    <>
      <PageTitle
        eyebrow="01 / DATA QUALITY"
        title="Trust starts at the source."
        description="Inspect the weekly contract, recorded repairs and issues that need investigation."
      />
      <ErrorNotice error={quality.error} />
      {quality.loading && <Loading />}
      {!dataset ? (
        <Empty description="Choose a project with a dataset to inspect its inputs." />
      ) : (
        <>
          <div className="stats-grid">
            <Stat
              label="Quality gate"
              value={<Badge value={dataset.quality_status} />}
              detail="Modeling requires no unresolved blockers"
            />
            <Stat
              label="Dataset version"
              value={`v${dataset.version}`}
              detail={dataset.source}
            />
            <Stat
              label="Observed from"
              value={
                <span className="date-stat">{date(dataset.date_start)}</span>
              }
              detail={`Through ${date(dataset.date_end)}`}
            />
            <Stat
              label="Raw observations"
              value={quality.data?.profile.row_count ?? "—"}
              detail="Before conservative normalization"
            />
          </div>
          <Panel
            title="Quality findings"
            subtitle="Repairs are recorded. Uncertain observations remain visible."
            action={
              <select
                aria-label="Filter quality findings"
                value={filter}
                onChange={(event) => setFilter(event.target.value)}
              >
                {["ALL", "BLOCKER", "WARNING", "RESOLVED"].map((value) => (
                  <option key={value}>{value}</option>
                ))}
              </select>
            }
          >
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Finding</th>
                    <th>Status</th>
                    <th>What happened</th>
                    <th>Affected weeks</th>
                  </tr>
                </thead>
                <tbody>
                  {quality.data?.issues
                    .filter(
                      (issue) => filter === "ALL" || issue.status === filter,
                    )
                    .map((issue, index) => (
                      <tr key={index}>
                        <td className="capitalize">
                          {issue.code.replaceAll("_", " ")}
                        </td>
                        <td>
                          <Badge value={issue.status} />
                        </td>
                        <td className="wide-cell">{issue.message}</td>
                        <td>
                          {issue.weeks.length ? (
                            <details>
                              <summary>{issue.weeks.length} listed</summary>
                              <div>
                                {issue.weeks.map((week) => (
                                  <div key={week}>{week}</div>
                                ))}
                              </div>
                            </details>
                          ) : (
                            "Not period-specific"
                          )}
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </Panel>
          <div className="two-column">
            <Panel
              title="Missing observations"
              subtitle="Raw column missingness"
            >
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Field</th>
                      <th>Missing values</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(
                      quality.data?.profile.missingness || {},
                    ).map(([field, count]) => (
                      <tr key={field}>
                        <td>{field}</td>
                        <td>{count}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Panel>
            <Panel
              title="Dataset provenance"
              subtitle="A stable identity for every downstream run"
            >
              <dl className="key-values">
                <dt>Dataset</dt>
                <dd className="mono">{dataset.id}</dd>
                <dt>Content hash</dt>
                <dd className="mono">{dataset.hash}</dd>
                <dt>Source</dt>
                <dd>{dataset.source}</dd>
              </dl>
              <EvidenceDetails
                data={quality.data}
                title="Inspect full quality report"
              />
            </Panel>
          </div>
        </>
      )}
    </>
  );
}
