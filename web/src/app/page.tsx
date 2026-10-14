"use client";
import Link from "next/link";
import { ArrowRight, ArrowUpRight, ShieldCheck } from "lucide-react";
import { useWorkspace } from "@/components/workspace";
import {
  Badge,
  Empty,
  ErrorNotice,
  Loading,
  PageTitle,
  Panel,
  Stat,
} from "@/components/ui";
import { PredictionChart } from "@/components/prediction-chart";
import { date, money, percent, useResource } from "@/lib/api";
import type { Comparison, Predictions, RunPage } from "@/lib/types";

export default function Overview() {
  const { project, model, dataset, health, healthError, revision } =
    useWorkspace();
  const predictions = useResource<Predictions>(
    model?.status === "SUCCEEDED"
      ? `/model-runs/${model.id}/predictions?limit=520`
      : null,
    revision,
  );
  const comparison = useResource<Comparison>(
    project ? `/projects/${project.id}/model-comparison` : null,
    revision,
  );
  const history = useResource<RunPage>(
    project ? `/projects/${project.id}/runs?limit=4` : null,
    revision,
  );
  const metrics = comparison.data?.items.find(
    (item) => item.model_run_id === model?.id,
  )?.posterior_predictive?.holdout;
  return (
    <>
      <PageTitle
        eyebrow="THE BIG PICTURE"
        title="Clarity before your next move."
        description="Your marketing performance, model health and decision readiness, together."
      >
        <span className="date-chip">
          {dataset
            ? `${date(dataset.date_start)} — ${date(dataset.date_end)}`
            : "Weekly observations"}
        </span>
      </PageTitle>
      <div className="stats-grid">
        <Stat
          label="Decision readiness"
          value={health ? <Badge value={health.decision_status} /> : "—"}
          detail="Current evidence, not job status"
        />
        <Stat
          label="Holdout prediction error"
          value={money(metrics?.metrics.mae)}
          detail="Mean absolute error · AUD / week"
        />
        <Stat
          label="Holdout interval coverage"
          value={percent(metrics?.coverage_90)}
          detail="Observed coverage of the 90% interval"
        />
        <Stat
          label="Training observations"
          value={model?.training_window?.rows ?? "—"}
          detail={
            model
              ? `${model.configuration.channels.length} media channels · weekly`
              : "Select a completed model"
          }
        />
      </div>
      <ErrorNotice error={healthError || comparison.error} />
      <div className="overview-grid">
        <Panel
          title="Revenue, observed and modeled"
          subtitle="Weekly revenue · AUD · conditional on realized spend and controls"
          action={
            <Link className="text-link" href="/model-lab">
              Model Lab <ArrowUpRight size={14} />
            </Link>
          }
        >
          {predictions.loading ? (
            <Loading />
          ) : predictions.error ? (
            <ErrorNotice error={predictions.error} />
          ) : predictions.data ? (
            <>
              <PredictionChart data={predictions.data} />
              <div className="chart-footnote">
                Dashed bounds show posterior uncertainty. Shaded weeks were held
                out from fitting.
              </div>
            </>
          ) : (
            <Empty
              description="Train a model to see observed revenue, predictions and uncertainty."
              href="/model-lab"
            />
          )}
        </Panel>
        <section className="decision-panel">
          <div className="eyebrow">
            <ShieldCheck size={15} /> DECISION CHECK
          </div>
          <h2>
            {health?.decision_status === "BLOCK"
              ? "Good predictions need strong evidence."
              : "Confidence starts with validation."}
          </h2>
          <p>
            {health?.decision_status === "BLOCK"
              ? "This model cannot support a budget recommendation yet. Review the evidence before changing your plan."
              : "Predictive fit is one piece of the picture. Release checks and decision stability determine what you can recommend."}
          </p>
          <div className="check-list">
            <div>
              <span>Data quality</span>
              {dataset ? (
                <Badge value={dataset.quality_status} />
              ) : (
                <span>Not available</span>
              )}
            </div>
            <div>
              <span>Model release</span>
              {health ? (
                <Badge value={health.decision_status} />
              ) : (
                <span>Not evaluated</span>
              )}
            </div>
            <div>
              <span>Decision stability</span>
              <span className="muted">Scenario required</span>
            </div>
          </div>
          <Link className="button" href="/model-health">
            Review model health
            <ArrowRight size={16} />
          </Link>
        </section>
      </div>
      <div className="two-column">
        <Panel
          title="What needs your attention"
          subtitle="Reasons returned by the release policy"
          action={<span className="section-index">01</span>}
        >
          {health?.policy?.reasons.length ? (
            <div className="reason-list">
              {health.policy.reasons.slice(0, 4).map((reason, index) => (
                <div className="reason" key={index}>
                  <span
                    className={`reason-dot ${reason.state === "BLOCK" ? "red" : "amber"}`}
                  />
                  <div>
                    <strong>{reason.code.replaceAll("_", " ")}</strong>
                    <p>{reason.message}</p>
                  </div>
                  <Badge value={reason.state} />
                </div>
              ))}
            </div>
          ) : (
            <Empty
              title={
                health?.policy
                  ? "No release issues recorded"
                  : "Awaiting evaluation"
              }
              description="Completed validation will explain which decisions the evidence supports."
            />
          )}
        </Panel>
        <Panel
          title="Latest activity"
          subtitle="An auditable path from data to decisions"
          action={
            <Link className="text-link" href="/runs">
              All runs <ArrowUpRight size={14} />
            </Link>
          }
        >
          {history.loading ? (
            <Loading />
          ) : history.error ? (
            <ErrorNotice error={history.error} />
          ) : history.data?.items.length ? (
            <div className="activity-list">
              {history.data.items.map((run) => (
                <Link
                  href="/runs"
                  className="activity"
                  key={`${run.kind}-${run.id}`}
                >
                  <span className="activity-icon">
                    <ArrowUpRight size={16} />
                  </span>
                  <div>
                    <strong>
                      {run.kind[0].toUpperCase() + run.kind.slice(1)} run
                    </strong>
                    <small>
                      {date(run.created_at)} · {run.id.slice(0, 8)}
                    </small>
                  </div>
                  <Badge value={run.status} />
                </Link>
              ))}
            </div>
          ) : (
            <Empty
              title="A fresh start"
              description="Model and evaluation runs will appear here with their provenance."
            />
          )}
        </Panel>
      </div>
    </>
  );
}
