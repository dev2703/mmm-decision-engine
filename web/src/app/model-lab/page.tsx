"use client";
import { useState } from "react";
import { Plus } from "lucide-react";
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
} from "@/components/ui";
import { PredictionChart } from "@/components/prediction-chart";
import { Chart, axis } from "@/components/chart";
import {
  api,
  channelName,
  money,
  number,
  percent,
  useResource,
} from "@/lib/api";
import type { Channel, Comparison, Predictions } from "@/lib/types";

export default function ModelLab() {
  const {
    project,
    model,
    models,
    dataset,
    revision,
    readOnly,
    selectModel,
    refresh,
  } = useWorkspace();
  const comparison = useResource<Comparison>(
    project ? `/projects/${project.id}/model-comparison` : null,
    revision,
  );
  const predictions = useResource<Predictions>(
    model?.status === "SUCCEEDED"
      ? `/model-runs/${model.id}/predictions?limit=520`
      : null,
    revision,
  );
  const [channel, setChannel] = useState("");
  const selectedChannel = model?.configuration.channels.includes(channel)
    ? channel
    : model?.configuration.channels[0];
  const evidence = useResource<Channel>(
    model?.status === "SUCCEEDED" && selectedChannel
      ? `/model-runs/${model.id}/channels/${encodeURIComponent(selectedChannel)}`
      : null,
    revision,
  );
  const [showForm, setShowForm] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error>();
  async function train(form: FormData) {
    if (!project || readOnly) return;
    setBusy(true);
    setError(undefined);
    try {
      const experiment = await api<{ id: string }>(
        `/projects/${project.id}/experiments`,
        {
          method: "POST",
          body: JSON.stringify({
            dataset_id: dataset?.id,
            hypothesis: form.get("hypothesis"),
            configuration: {
              draws: Number(form.get("draws")),
              tune: Number(form.get("tune")),
              chains: 4,
              seed: 42,
            },
          }),
        },
      );
      const run = await api<{ id: string }>(
        `/experiments/${experiment.id}/run`,
        { method: "POST" },
      );
      selectModel(run.id);
      setShowForm(false);
      refresh();
    } catch (failure) {
      setError(failure as Error);
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <PageTitle
        eyebrow="02 / MODEL LAB"
        title="Understand what the model learned."
        description="Compare like-for-like candidates, inspect fit and explore channel uncertainty."
      >
        <button
          className="button secondary"
          onClick={() => setShowForm((value) => !value)}
          disabled={!project || readOnly}
        >
          <Plus size={16} />
          New experiment
        </button>
      </PageTitle>
      <ErrorNotice error={comparison.error || error} />
      {showForm && (
        <Panel
          title="New Bayesian experiment"
          subtitle="Training queues a candidate. Evaluation is still required before any budget decision."
        >
          <form className="form-grid" action={train}>
            <label className="full-width">
              Hypothesis
              <input
                name="hypothesis"
                required
                minLength={5}
                maxLength={2000}
                placeholder="What change are you testing?"
              />
            </label>
            <label>
              Draws per chain
              <input
                name="draws"
                type="number"
                min="10"
                max="10000"
                defaultValue="500"
                required
              />
            </label>
            <label>
              Tuning steps
              <input
                name="tune"
                type="number"
                min="10"
                max="20000"
                defaultValue="1000"
                required
              />
            </label>
            <div className="full-width form-actions">
              <span className="muted">
                4 chains · seed 42 · model fitting runs as a separate job
              </span>
              <button className="button" disabled={busy}>
                {busy ? "Queuing…" : "Queue training"}
              </button>
            </div>
          </form>
        </Panel>
      )}
      <Panel
        title="Candidate comparison"
        subtitle="Matching data, training windows and prediction context are required for a fair comparison."
      >
        {comparison.loading ? (
          <Loading />
        ) : comparison.data?.items.length ? (
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Candidate</th>
                  <th>Run status</th>
                  <th>Training window</th>
                  <th>Holdout MAE</th>
                  <th>Holdout WAPE</th>
                  <th>90% coverage</th>
                </tr>
              </thead>
              <tbody>
                {comparison.data.items.map((item) => (
                  <tr
                    key={item.model_run_id}
                    className={
                      item.model_run_id === model?.id ? "selected-row" : ""
                    }
                  >
                    <td>
                      <button
                        className="text-link"
                        onClick={() => selectModel(item.model_run_id)}
                      >
                        {item.model_run_id.slice(0, 8)}
                        {item.model_run_id === model?.id && (
                          <span className="selected-dot" />
                        )}
                      </button>
                    </td>
                    <td>
                      <Badge value={item.job_status} />
                    </td>
                    <td>
                      {item.training_window ? (
                        <>
                          {item.training_window.start}
                          <small className="block">
                            to {item.training_window.end}
                          </small>
                        </>
                      ) : (
                        "—"
                      )}
                    </td>
                    <td>
                      {money(item.posterior_predictive?.holdout?.metrics.mae)}
                    </td>
                    <td>
                      {percent(
                        item.posterior_predictive?.holdout?.metrics.wape,
                      )}
                    </td>
                    <td>
                      {percent(item.posterior_predictive?.holdout?.coverage_90)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Empty description="Create an experiment and train a candidate to begin comparison." />
        )}
        {comparison.data &&
          comparison.data.total > comparison.data.items.length && (
            <p className="chart-footnote">
              Showing the latest {comparison.data.items.length} of{" "}
              {comparison.data.total} candidates. All jobs remain available in
              Runs.
            </p>
          )}
      </Panel>
      <Notice title="Fit is not a release decision">
        These are conditional predictions using realized spend and controls.
        Accuracy alone does not establish causal validity or authorize
        optimization.
      </Notice>
      <div className="two-column">
        <Panel
          title="Actual vs predicted"
          subtitle="AUD / week · 90% posterior interval"
        >
          {predictions.loading ? (
            <Loading />
          ) : predictions.error ? (
            <ErrorNotice error={predictions.error} />
          ) : predictions.data ? (
            <PredictionChart data={predictions.data} />
          ) : (
            <Empty description="A completed model is required." />
          )}
        </Panel>
        <Panel
          title="Residuals through time"
          subtitle="Actual revenue minus predicted revenue · AUD / week"
        >
          {predictions.data ? (
            <PredictionChart data={predictions.data} residuals />
          ) : (
            <Empty description="Residual evidence appears after fitting." />
          )}
        </Panel>
      </div>
      <Panel
        title="Channel response & uncertainty"
        subtitle="Conditional saturation response to adstocked spend"
        action={
          <select
            aria-label="Channel"
            value={selectedChannel || ""}
            onChange={(event) => setChannel(event.target.value)}
          >
            {model?.configuration.channels.map((item) => (
              <option key={item} value={item}>
                {channelName(item)}
              </option>
            ))}
          </select>
        }
      >
        <ErrorNotice error={evidence.error} />
        {evidence.loading ? (
          <Loading />
        ) : evidence.data ? (
          <div className="channel-grid">
            <div>
              <Chart
                label="Conditional saturation response with 90% credible bounds"
                option={{
                  xAxis: {
                    ...axis,
                    type: "value",
                    name: "Adstocked spend (AUD)",
                    nameLocation: "middle",
                    nameGap: 30,
                  },
                  yAxis: { ...axis, type: "value", name: "AUD / week" },
                  series: [
                    {
                      name: "Median response",
                      type: "line",
                      symbol: "none",
                      data: evidence.data.response_curve.map((point) => [
                        point.adstocked_spend,
                        point.median,
                      ]),
                    },
                    ...(["lower_90", "upper_90"] as const).map((bound) => ({
                      name: bound === "lower_90" ? "90% lower" : "90% upper",
                      type: "line",
                      symbol: "none",
                      lineStyle: { type: "dashed", width: 1 },
                      data: evidence.data!.response_curve.map((point) => [
                        point.adstocked_spend,
                        point[bound],
                      ]),
                    })),
                  ],
                }}
              />
            </div>
            <div className="channel-summary">
              <div className="eyebrow">
                {channelName(evidence.data.channel)}
              </div>
              <h3>
                {number(evidence.data.summary.roi_mean)}×{" "}
                <small>posterior mean ROI</small>
              </h3>
              <p>
                90% interval: {number(evidence.data.summary.roi_interval_90[0])}
                –{number(evidence.data.summary.roi_interval_90[1])} AUD / AUD
              </p>
              <hr />
              <strong>{money(evidence.data.summary.contribution_mean)}</strong>
              <p>Modeled contribution over the training window.</p>
              <p className="muted">
                90% interval:{" "}
                {money(evidence.data.summary.contribution_interval_90[0])}–
                {money(evidence.data.summary.contribution_interval_90[1])}
              </p>
            </div>
          </div>
        ) : (
          <Empty description="Channel evidence is available for completed, intact models." />
        )}
      </Panel>
      {predictions.data && (
        <EvidenceDetails
          data={predictions.data.provenance}
          title="Inspect training provenance and limitations"
        />
      )}
      {models.length === 0 && (
        <p className="muted">
          All model and release results are produced by the registered
          scientific workflow.
        </p>
      )}
    </>
  );
}
