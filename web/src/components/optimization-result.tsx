"use client";
import { Badge, Empty, EvidenceDetails, Notice, Panel, Stat } from "./ui";
import { Chart, axis } from "./chart";
import { channelName, money, number, percent } from "@/lib/api";
import type { Optimization } from "@/lib/types";
export function OptimizationResult({ result }: { result: Optimization }) {
  const summary = result.summary;
  return (
    <Panel
      title="Scenario outcome"
      subtitle={`Optimization ${result.id}`}
      action={<Badge value={result.status} />}
    >
      {!summary ? (
        <Empty
          title={
            result.status === "FAILED"
              ? "Optimization did not complete"
              : "Your scenario is queued"
          }
          description={
            result.status === "FAILED"
              ? "Inspect the run history for the failure. No recommendation has been produced."
              : "Results will appear after the registered job finishes. No allocation is recommended while it is pending."
          }
          href="/runs"
          action="View run history"
        />
      ) : (
        <>
          <div className="inset">
            <Notice
              positive={
                result.recommendation_allowed && result.evidence_current
              }
              title={
                result.recommendation_allowed && result.evidence_current
                  ? "Recommendation permitted by current evidence"
                  : "This result does not authorize a recommendation"
              }
            >
              {!result.evidence_current
                ? "Release evidence changed. This is a historical result; create a new scenario using current evidence."
                : "Review stability, downside and extrapolation. An unstable or unsupported allocation cannot authorize a recommendation."}
            </Notice>
          </div>
          <div className="stats-grid inset">
            <Stat
              label="Expected modeled outcome"
              value={money(
                summary.selected.expected_modeled_media_revenue,
                true,
              )}
              detail={`${summary.horizon_weeks} weeks · modeled media revenue`}
            />
            <Stat
              label="Probability of beating current"
              value={percent(summary.selected.probability_beat_current)}
              detail="Joint posterior draws"
            />
            <Stat
              label="Downside probability"
              value={percent(summary.selected.downside_probability)}
              detail="Modeled outcome below current plan"
            />
            <Stat
              label="Decision instability · P95"
              value={number(summary.selected.stability_p95, 3)}
              detail="L1 allocation difference / budget"
            />
          </div>
          <div className="inset muted">
            90% modeled media revenue interval:{" "}
            {money(summary.selected.media_revenue_interval_90[0])}–
            {money(summary.selected.media_revenue_interval_90[1])}. This
            excludes nonmedia revenue and observation noise.
          </div>
          <Chart
            label="Current versus proposed constant weekly budget by channel"
            option={{
              legend: { top: 5 },
              xAxis: {
                ...axis,
                type: "category",
                data: Object.keys(summary.baseline_allocation).map(channelName),
              },
              yAxis: { ...axis, type: "value", name: "AUD / week" },
              series: [
                {
                  name: "Current",
                  type: "bar",
                  data: Object.values(summary.baseline_allocation),
                  barMaxWidth: 35,
                },
                {
                  name: "Proposed",
                  type: "bar",
                  data: Object.keys(summary.baseline_allocation).map(
                    (channel) => summary.selected.allocation[channel],
                  ),
                  barMaxWidth: 35,
                },
              ],
            }}
          />
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Alternative</th>
                  <th>Expected improvement · AUD</th>
                  <th>Downside probability</th>
                  <th>Instability P95</th>
                  <th>Outside historical support</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(summary.alternatives).map(([name, value]) => (
                  <tr key={name}>
                    <td className="capitalize">{name}</td>
                    <td>{money(value.expected_improvement)}</td>
                    <td>{percent(value.downside_probability)}</td>
                    <td>{number(value.stability_p95, 3)}</td>
                    <td>
                      {value.extrapolated_channels
                        .map(channelName)
                        .join(", ") || "None recorded"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <EvidenceDetails
            data={result}
            title="Inspect complete scenario evidence"
          />
        </>
      )}
    </Panel>
  );
}
