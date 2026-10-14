export type Project = {
  id: string;
  name: string;
  current_dataset_id: string | null;
  created_at: string;
};
export type Dataset = {
  id: string;
  project_id: string;
  version: number;
  source: string;
  quality_status: string;
  date_start: string;
  date_end: string;
  hash: string;
};
export type Model = {
  id: string;
  experiment_id: string;
  dataset_id: string;
  status: string;
  model_status: string;
  created_at: string;
  configuration: {
    channels: string[];
    draws: number;
    chains: number;
    seed: number;
  };
  training_window: { start: string; end: string; rows: number } | null;
  code_version: string | null;
};
export type Reason = {
  code: string;
  state: string;
  message: string;
  channel: string | null;
  value: number | null;
};
export type Policy = {
  state: string;
  reasons: Reason[];
  channel_movement_limits: Record<string, number>;
  policy_version: string;
};
export type Health = {
  model_run_id: string;
  model_status: string;
  decision_status: string;
  diagnostics: Record<string, unknown> | null;
  policy: Policy | null;
  evaluation: {
    id: string;
    status: string;
    created_at: string;
    configuration: Record<string, unknown>;
  } | null;
};
export type Quality = {
  status: string;
  profile: {
    row_count: number;
    missingness: Record<string, number>;
    cardinality: Record<string, number>;
  };
  issues: { code: string; status: string; message: string; weeks: string[] }[];
};
export type Predictions = {
  units: string;
  provenance: {
    training_window: { start: string; end: string; rows: number };
    limitations: string[];
    dataset_hash: string;
    model_record_hash: string;
  };
  rows: {
    week: string;
    split: string;
    actual: number;
    predicted: number;
    lower_90: number;
    upper_90: number;
    residual: number;
  }[];
};
export type Channel = {
  channel: string;
  summary: {
    roi_mean: number;
    roi_interval_90: [number, number];
    contribution_mean: number;
    contribution_interval_90: [number, number];
  };
  contribution_units: string;
  roi_units: string;
  response_units: string;
  response_curve: {
    adstocked_spend: number;
    median: number;
    lower_90: number;
    upper_90: number;
  }[];
};
export type Comparison = {
  items: {
    model_run_id: string;
    dataset_id: string;
    dataset_hash: string | null;
    training_window: { start: string; end: string; rows: number } | null;
    job_status: string;
    prediction_context: string | null;
    model_record_hash: string | null;
    posterior_predictive: Record<
      string,
      { metrics: Record<string, number | null>; coverage_90: number }
    > | null;
  }[];
  total: number;
  next_offset: number | null;
};
export type Evaluation = {
  evaluation_run_id: string;
  record_hash: string;
  policy: Policy;
  metrics: {
    test: string;
    metric: string;
    value: number | null;
    passed: boolean | null;
    channel: string | null;
  }[];
};
export type Run = {
  id: string;
  kind: string;
  status: string;
  model_run_id: string;
  experiment_id: string;
  dataset_id: string;
  evaluation_run_id: string | null;
  scenario_id: string | null;
  record_hash: string | null;
  error_type: string | null;
  created_at: string;
  finished_at: string | null;
};
export type RunPage = { items: Run[]; next_offset: number | null };
export type Alternative = {
  allocation: Record<string, number>;
  expected_modeled_media_revenue: number;
  media_revenue_interval_90: [number, number];
  expected_improvement: number;
  improvement_p05: number;
  probability_beat_current: number;
  downside_probability: number;
  stability_median: number;
  stability_p95: number;
  extrapolated_channels: string[];
  direction_flip_probability: Record<string, number | null>;
  recommendation_allowed: boolean;
  decision_state: string;
};
export type Optimization = {
  id: string;
  scenario_id: string;
  status: string;
  evidence_current: boolean;
  recommendation_allowed: boolean;
  summary: {
    selected: Alternative;
    alternatives: Record<string, Alternative>;
    baseline_allocation: Record<string, number>;
    risk_policy: string;
    horizon_weeks: number;
    outcome_units: string;
  } | null;
};
export type AgentAnswer = {
  status: string;
  message: string;
  evidence: {
    id: string;
    tool: string;
    evidence: { source: string; data: unknown; decision_use: string } | null;
  }[];
  issues: { tool: string; status: string }[];
  trace: {
    request_id: string;
    model: string;
    tools: { tool: string; status: string; latency_ms: number }[];
    usage: {
      input_tokens: number | null;
      output_tokens: number | null;
      cost_usd: number | null;
    };
  };
};
