"use client";
import { useWorkspace } from "@/components/workspace";
import {
  Badge,
  Empty,
  ErrorNotice,
  EvidenceDetails,
  Loading,
  Notice,
  PageTitle,
  Panel,
  Stat,
} from "@/components/ui";
import { channelName, number, percent, useResource } from "@/lib/api";
import type { Evaluation } from "@/lib/types";

export default function ModelHealth() {
  const { model, health, healthError, revision } = useWorkspace();
  const evaluation = useResource<Evaluation>(
    health?.evaluation?.status === "SUCCEEDED"
      ? `/evaluation-runs/${health.evaluation.id}/evidence`
      : null,
    revision,
  );
  const diagnostic = (key: string) =>
    typeof health?.diagnostics?.[key] === "number"
      ? number(health.diagnostics[key] as number, 4)
      : "—";
  return (
    <>
      <PageTitle
        eyebrow="03 / MODEL HEALTH"
        title="Evidence earns permission."
        description="Understand the validation results, restrictions and uncertainty behind the release decision."
      />
      <ErrorNotice error={healthError || evaluation.error} />
      {!model ? (
        <Empty description="Select a model to inspect its current release authority." />
      ) : (
        <>
          <Notice
            positive={health?.decision_status === "PASS"}
            title={
              health?.decision_status === "BLOCK"
                ? "Budget recommendations are blocked"
                : health
                  ? `Current release: ${health.decision_status}`
                  : "Release evidence is unavailable"
            }
          >
            {health?.decision_status === "BLOCK"
              ? "Resolve the recorded validation failures before using this model to allocate budget. The decision gate cannot be overridden."
              : "Scenario feasibility, uncertainty and historical support still need to pass before an allocation can be recommended."}
          </Notice>
          <div className="stats-grid">
            <Stat
              label="Release state"
              value={health ? <Badge value={health.decision_status} /> : "—"}
              detail="Latest evaluation governs decisions"
            />
            <Stat
              label="Maximum R-hat"
              value={diagnostic("max_rhat")}
              detail="Convergence diagnostic"
            />
            <Stat
              label="Minimum bulk ESS"
              value={diagnostic("min_ess_bulk")}
              detail="Effective posterior sample size"
            />
            <Stat
              label="Divergences"
              value={diagnostic("divergences")}
              detail="Sampler geometry diagnostic"
            />
          </div>
          <Panel
            title="Release reasoning"
            subtitle="Unabridged reasons from the deterministic policy"
          >
            {health?.policy ? (
              <div className="reason-list">
                {health.policy.reasons.length ? (
                  health.policy.reasons.map((reason, index) => (
                    <div className="reason" key={index}>
                      <Badge value={reason.state} />
                      <div>
                        <strong>
                          {reason.code.replaceAll("_", " ")}
                          {reason.channel &&
                            ` · ${channelName(reason.channel)}`}
                        </strong>
                        <p>{reason.message}</p>
                      </div>
                      <span className="numeric">
                        {reason.value == null ? "" : number(reason.value, 4)}
                      </span>
                    </div>
                  ))
                ) : (
                  <p className="inset muted">No policy failures recorded.</p>
                )}
              </div>
            ) : (
              <Empty
                title="Evaluation required"
                description={
                  health?.evaluation
                    ? `The latest evaluation is ${health.evaluation.status.toLowerCase()}. An older release is not reused.`
                    : "No completed evaluation is registered for this model."
                }
              />
            )}
          </Panel>
          <Panel
            title="Validation evidence"
            subtitle="Raw metric values retain upstream definitions and units. Missing results stay missing."
          >
            {evaluation.loading ? (
              <Loading />
            ) : evaluation.data ? (
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Test</th>
                      <th>Metric</th>
                      <th>Channel</th>
                      <th>Source value</th>
                      <th>Source check</th>
                    </tr>
                  </thead>
                  <tbody>
                    {evaluation.data.metrics.map((metric, index) => (
                      <tr key={index}>
                        <td className="capitalize">
                          {metric.test.replaceAll("_", " ")}
                        </td>
                        <td>{metric.metric.replaceAll("_", " ")}</td>
                        <td>
                          {metric.channel
                            ? channelName(metric.channel)
                            : "All / not channel-specific"}
                        </td>
                        <td>{number(metric.value, 5)}</td>
                        <td>
                          {metric.passed == null ? (
                            "Unavailable"
                          ) : (
                            <Badge value={metric.passed ? "PASS" : "FAILED"} />
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <Empty description="The latest completed evaluation supplies these checks." />
            )}
          </Panel>
          {!!Object.keys(health?.policy?.channel_movement_limits || {})
            .length && (
            <Panel
              title="Channel restrictions"
              subtitle="These limits are enforced in every scenario"
            >
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Channel</th>
                      <th>Maximum movement</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(
                      health!.policy!.channel_movement_limits,
                    ).map(([channel, limit]) => (
                      <tr key={channel}>
                        <td>{channelName(channel)}</td>
                        <td>{percent(limit)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Panel>
          )}
          <EvidenceDetails
            data={health}
            title="Inspect full health and evaluation identity"
          />
        </>
      )}
    </>
  );
}
