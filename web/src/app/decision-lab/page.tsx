"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowRight, LockKeyhole, Play } from "lucide-react";
import { useWorkspace } from "@/components/workspace";
import {
  Badge,
  Empty,
  ErrorNotice,
  EvidenceDetails,
  Notice,
  PageTitle,
  Panel,
} from "@/components/ui";
import { api, channelName, money } from "@/lib/api";
import { OptimizationResult } from "@/components/optimization-result";
import type { Model, Optimization } from "@/lib/types";

type BudgetRow = {
  current: string;
  floor: string;
  cap: string;
  movement: string;
  protected: string;
};
type Validation = {
  valid: boolean;
  effective_bounds: Record<string, [number, number]>;
  release: { state: string };
  evaluation_run_id: string;
};

export default function DecisionLab() {
  const { model, health, healthError, readOnly } = useWorkspace();
  const allowed = !!health && health.decision_status !== "BLOCK" && !readOnly;
  return (
    <>
      <PageTitle
        eyebrow="04 / DECISION LAB"
        title="Explore the tradeoffs."
        description="Set real business constraints, compare risk preferences and see what the evidence supports."
      />
      <ErrorNotice error={healthError} />
      {!allowed && (
        <Notice
          title={
            readOnly
              ? "This workspace is read-only"
              : "Scenario execution is unavailable"
          }
        >
          {health?.decision_status === "BLOCK"
            ? "The selected model is blocked. You can review inputs here, but validation and optimization require a permitted release."
            : "Select a model with current, verified release evidence to explore allocations."}{" "}
          <Link className="text-link" href="/model-health">
            Review model health <ArrowRight size={13} />
          </Link>
        </Notice>
      )}
      {model ? (
        <ScenarioPlanner key={model.id} model={model} allowed={allowed} />
      ) : (
        <Empty description="Select a model before defining budget constraints." />
      )}
    </>
  );
}

function ScenarioPlanner({
  model,
  allowed,
}: {
  model: Model;
  allowed: boolean;
}) {
  const [rows, setRows] = useState<Record<string, BudgetRow>>(() =>
    Object.fromEntries(
      model.configuration.channels.map((channel) => [
        channel,
        { current: "", floor: "", cap: "", movement: "", protected: "" },
      ]),
    ),
  );
  const [total, setTotal] = useState("");
  const [horizon, setHorizon] = useState("13");
  const [risk, setRisk] = useState("conservative");
  const [validated, setValidated] = useState<Validation>();
  const [scenarioId, setScenarioId] = useState<string>();
  const [result, setResult] = useState<Optimization>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error>();
  const invalidate = () => {
    setValidated(undefined);
    setScenarioId(undefined);
    setResult(undefined);
    setError(undefined);
  };
  function body() {
    const map = (field: keyof BudgetRow, divisor = 1) =>
      Object.fromEntries(
        Object.entries(rows)
          .filter(([, row]) => row[field] !== "")
          .map(([channel, row]) => [channel, Number(row[field]) / divisor]),
      );
    return {
      total: Number(total),
      current: map("current"),
      floors: map("floor"),
      caps: map("cap"),
      max_movement: map("movement", 100),
      protected_spend: map("protected"),
      horizon_weeks: Number(horizon),
      risk_policy: risk,
    };
  }
  async function validate() {
    if (!allowed) return;
    setBusy(true);
    setError(undefined);
    setValidated(undefined);
    try {
      setValidated(
        await api<Validation>(`/model-runs/${model.id}/constraints/validate`, {
          method: "POST",
          body: JSON.stringify(body()),
        }),
      );
    } catch (failure) {
      setError(failure as Error);
    } finally {
      setBusy(false);
    }
  }
  async function queue() {
    if (!allowed || !validated) return;
    setBusy(true);
    setError(undefined);
    try {
      const id =
        scenarioId ||
        (
          await api<{ id: string }>(`/model-runs/${model.id}/scenarios`, {
            method: "POST",
            body: JSON.stringify(body()),
          })
        ).id;
      setScenarioId(id);
      setResult(
        await api<Optimization>(`/scenarios/${id}/optimize`, {
          method: "POST",
          body: "{}",
        }),
      );
    } catch (failure) {
      setError(failure as Error);
      setValidated(undefined);
    } finally {
      setBusy(false);
    }
  }
  useEffect(() => {
    if (!result || !["QUEUED", "RUNNING"].includes(result.status)) return;
    const controller = new AbortController();
    const timer = setTimeout(() => {
      api<Optimization>(`/optimization-runs/${result.id}`, {
        signal: controller.signal,
      })
        .then((value) => {
          if (!controller.signal.aborted) setResult(value);
        })
        .catch((failure) => {
          if (!controller.signal.aborted) setError(failure);
        });
    }, 3000);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [result]);
  return (
    <>
      <Panel
        title="Your planning constraints"
        subtitle="Constant weekly spend in AUD. Blank optional bounds use the model policy."
      >
        <form
          onSubmit={(event) => {
            event.preventDefault();
            void validate();
          }}
        >
          <fieldset disabled={busy || !!result}>
            <div className="form-grid planning-fields">
              <label>
                Total weekly budget (AUD)
                <input
                  aria-label="Total weekly budget"
                  type="number"
                  min="0.01"
                  step="0.01"
                  required
                  value={total}
                  onChange={(event) => {
                    setTotal(event.target.value);
                    invalidate();
                  }}
                  placeholder="Enter your budget"
                />
              </label>
              <label>
                Planning horizon
                <select
                  value={horizon}
                  onChange={(event) => {
                    setHorizon(event.target.value);
                    invalidate();
                  }}
                >
                  {[4, 13, 26, 52].map((weeks) => (
                    <option value={weeks} key={weeks}>
                      {weeks} weeks
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Risk preference
                <select
                  value={risk}
                  onChange={(event) => {
                    setRisk(event.target.value);
                    invalidate();
                  }}
                >
                  <option value="conservative">
                    Conservative · downside-aware
                  </option>
                  <option value="expected">Expected modeled outcome</option>
                </select>
              </label>
            </div>
            <div className="table-scroll">
              <table className="budget-table">
                <thead>
                  <tr>
                    <th>Channel</th>
                    <th>Current / week</th>
                    <th>Floor</th>
                    <th>Cap</th>
                    <th>Max move %</th>
                    <th>Protected spend</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(rows).map(([channel, row]) => (
                    <tr key={channel}>
                      <td>
                        <strong>{channelName(channel)}</strong>
                      </td>
                      {(
                        [
                          "current",
                          "floor",
                          "cap",
                          "movement",
                          "protected",
                        ] as const
                      ).map((field) => (
                        <td key={field}>
                          <input
                            type="number"
                            aria-label={`${channelName(channel)} ${field}`}
                            required={field === "current"}
                            min="0"
                            max={field === "movement" ? "100" : undefined}
                            step="any"
                            value={row[field]}
                            placeholder={
                              field === "current" ? "Required" : "Optional"
                            }
                            onChange={(event) => {
                              setRows((value) => ({
                                ...value,
                                [channel]: {
                                  ...value[channel],
                                  [field]: event.target.value,
                                },
                              }));
                              invalidate();
                            }}
                          />
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </fieldset>
          <div className="form-actions inset">
            <span className="muted">
              <LockKeyhole size={14} /> Release restrictions always apply.
            </span>
            <button
              className="button secondary"
              disabled={busy || !allowed || !!result}
            >
              {busy ? "Checking…" : "Validate constraints"}
            </button>
          </div>
        </form>
      </Panel>
      <ErrorNotice error={error} />
      {validated && !result && (
        <Panel
          title="Review the feasible plan"
          subtitle="Feasibility is not a recommendation. Current release is rechecked when queuing."
        >
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Channel</th>
                  <th>Effective floor / week</th>
                  <th>Effective cap / week</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(validated.effective_bounds).map(
                  ([channel, bounds]) => (
                    <tr key={channel}>
                      <td>{channelName(channel)}</td>
                      <td>{money(bounds[0])}</td>
                      <td>{money(bounds[1])}</td>
                    </tr>
                  ),
                )}
              </tbody>
            </table>
          </div>
          <div className="form-actions inset">
            <Badge value={validated.release.state} />
            <button
              className="button"
              onClick={queue}
              disabled={busy || !allowed}
            >
              <Play size={15} />
              Create and queue scenario
            </button>
          </div>
        </Panel>
      )}
      {result && <OptimizationResult result={result} />}
      {result && (
        <button className="button secondary" onClick={invalidate}>
          Start another scenario
        </button>
      )}
    </>
  );
}
