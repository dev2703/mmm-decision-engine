"use client";

import { createContext, useContext, useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Activity,
  ArrowUpRight,
  ChartNoAxesCombined,
  ChevronDown,
  Database,
  FlaskConical,
  LayoutDashboard,
  Menu,
  MessageSquare,
  RefreshCw,
  ShieldCheck,
  SlidersHorizontal,
  X,
} from "lucide-react";
import { api, useResource } from "@/lib/api";
import type { Dataset, Health, Model, Project } from "@/lib/types";
import { Badge, ErrorNotice } from "./ui";

const navigation = [
  ["/", "Overview", LayoutDashboard],
  ["/data-quality", "Data Quality", Database],
  ["/model-lab", "Model Lab", FlaskConical],
  ["/model-health", "Model Health", ShieldCheck],
  ["/decision-lab", "Decision Lab", SlidersHorizontal],
  ["/ask", "Ask DecisionGuard", MessageSquare],
  ["/runs", "Runs & history", Activity],
] as const;
type WorkspaceState = {
  project?: Project;
  projects: Project[];
  model?: Model;
  models: Model[];
  dataset?: Dataset;
  health?: Health;
  healthError?: Error;
  loading: boolean;
  revision: number;
  readOnly: boolean;
  agentEnabled: boolean;
  selectProject: (id: string) => void;
  selectModel: (id: string) => void;
  refresh: () => void;
};
const Context = createContext<WorkspaceState | null>(null);
export function useWorkspace() {
  const value = useContext(Context);
  if (!value) throw new Error("Workspace missing");
  return value;
}

export function Workspace({ children }: { children: React.ReactNode }) {
  const path = usePathname();
  const [open, setOpen] = useState(false);
  const [revision, setRevision] = useState(0);
  const [projectId, setProjectId] = useState("");
  const [modelId, setModelId] = useState("");
  const [creating, setCreating] = useState(false);
  const [actionError, setActionError] = useState<Error>();
  const projects = useResource<Project[]>("/projects", revision);
  const capabilities = useResource<{
    read_only: boolean;
    agent_enabled: boolean;
  }>("/capabilities", revision);
  const project =
    projects.data?.find((item) => item.id === projectId) || projects.data?.[0];
  const models = useResource<Model[]>(
    project ? `/projects/${project.id}/model-runs` : null,
    revision,
  );
  const model =
    models.data?.find((item) => item.id === modelId) ||
    models.data?.find((item) => item.status === "SUCCEEDED") ||
    models.data?.[0];
  const datasets = useResource<Dataset[]>(
    project ? `/projects/${project.id}/datasets` : null,
    revision,
  );
  const dataset = datasets.data?.find(
    (item) => item.id === (model?.dataset_id || project?.current_dataset_id),
  );
  const health = useResource<Health>(
    model ? `/model-runs/${model.id}/health` : null,
    revision,
  );
  const refresh = () => setRevision((value) => value + 1);
  const selectProject = (id: string) => {
    setProjectId(id);
    setModelId("");
  };
  useEffect(() => setOpen(false), [path]);
  async function createWorkspace() {
    setCreating(true);
    setActionError(undefined);
    try {
      const created = await api<Project>("/projects", {
        method: "POST",
        body: JSON.stringify({ name: "Marketing workspace" }),
      });
      await api(`/projects/${created.id}/datasets/generate`, {
        method: "POST",
        body: JSON.stringify({ seed: 42, weeks: 156 }),
      });
      selectProject(created.id);
      refresh();
    } catch (error) {
      setActionError(error as Error);
    } finally {
      setCreating(false);
    }
  }
  const value: WorkspaceState = {
    project,
    projects: projects.data || [],
    model,
    models: models.data || [],
    dataset,
    health: health.data,
    healthError: health.error,
    loading: projects.loading || models.loading,
    revision,
    readOnly: capabilities.data?.read_only ?? true,
    agentEnabled: capabilities.data?.agent_enabled ?? false,
    selectProject,
    selectModel: setModelId,
    refresh,
  };
  return (
    <Context.Provider value={value}>
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <aside
        className={`sidebar ${open ? "open" : ""}`}
        aria-label="Main navigation"
      >
        <Link href="/" className="brand">
          <span className="brand-mark">
            <ChartNoAxesCombined size={21} />
          </span>
          <span>
            Decision<span className="brand-light">Guard</span>
            <small>MARKETING INTELLIGENCE</small>
          </span>
        </Link>
        <button
          className="icon-button mobile-close"
          aria-label="Close navigation"
          onClick={() => setOpen(false)}
        >
          <X size={20} />
        </button>
        <div className="workspace-label">WORKSPACE</div>
        <div className="workspace-select">
          <span className="workspace-avatar">M</span>
          <select
            aria-label="Project"
            value={project?.id || ""}
            onChange={(event) => selectProject(event.target.value)}
          >
            <option value="" disabled>
              Select a project
            </option>
            {projects.data?.map((item) => (
              <option key={item.id} value={item.id}>
                {item.name}
              </option>
            ))}
          </select>
          <ChevronDown size={14} />
        </div>
        <nav>
          {navigation.map(([href, name, Icon]) => (
            <Link
              key={href}
              href={href}
              aria-current={path === href ? "page" : undefined}
              className={path === href ? "active" : ""}
            >
              <Icon size={18} strokeWidth={1.7} />
              {name}
              {path === href && <span className="nav-dot" />}
            </Link>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="evidence-first">
            <ShieldCheck size={20} />
            <strong>Evidence before action</strong>
            <p>Every decision starts with a model you can trust.</p>
          </div>
          <div className="user-row">
            <span className="user-avatar">DG</span>
            <div>
              Research workspace<small>Weekly · AUD</small>
            </div>
          </div>
        </div>
      </aside>
      {open && (
        <button
          className="sidebar-backdrop"
          aria-label="Close navigation"
          onClick={() => setOpen(false)}
        />
      )}
      <div className="main-shell">
        <header className="topbar">
          <div className="breadcrumbs">
            <button
              className="icon-button mobile-menu"
              aria-label="Open navigation"
              onClick={() => setOpen(true)}
            >
              <Menu size={20} />
            </button>
            <span>Workspace</span>
            <span className="slash">/</span>
            <strong>
              {navigation.find((item) => item[0] === path)?.[1] || "Overview"}
            </strong>
          </div>
          <div className="topbar-actions">
            <span className="live-indicator">
              <span />
              {capabilities.data ? "Backend connected" : "Connecting"}
            </span>
            <button
              className="icon-button"
              aria-label="Refresh evidence"
              onClick={refresh}
            >
              <RefreshCw size={16} />
            </button>
          </div>
        </header>
        <main id="main">
          <ErrorNotice
            error={
              projects.error ||
              models.error ||
              datasets.error ||
              capabilities.error ||
              actionError
            }
          />
          {project && (
            <div className="model-toolbar">
              <label>
                Selected model{" "}
                <select
                  aria-label="Selected model"
                  value={model?.id || ""}
                  onChange={(event) => setModelId(event.target.value)}
                >
                  <option value="" disabled>
                    {models.loading
                      ? "Loading models…"
                      : "No model trained yet"}
                  </option>
                  {models.data?.map((item, index) => (
                    <option key={item.id} value={item.id}>
                      MMM {models.data!.length - index} · {item.id.slice(0, 8)}{" "}
                      · {item.status.toLowerCase()}
                    </option>
                  ))}
                </select>
              </label>
              {health.data && <Badge value={health.data.decision_status} />}
              <span className="toolbar-note">
                {capabilities.data?.read_only
                  ? "Read-only workspace"
                  : "Verified model evidence"}
              </span>
            </div>
          )}
          {!projects.loading && !projects.error && !project && (
            <div className="welcome">
              <div className="eyebrow">YOUR FIRST WORKSPACE</div>
              <h2>A clear view starts with real data.</h2>
              <p>
                Create a seeded synthetic dataset to explore data quality. Train
                and evaluate a model before making decisions.
              </p>
              <button
                className="button"
                disabled={
                  creating || capabilities.data?.read_only || !capabilities.data
                }
                onClick={createWorkspace}
              >
                {creating ? "Creating workspace…" : "Create sample workspace"}
                <ArrowUpRight size={16} />
              </button>
            </div>
          )}
          {children}
        </main>
        <footer className="footer">
          <span>DecisionGuard</span>
          <span>Predictive performance is not causal validity.</span>
        </footer>
      </div>
    </Context.Provider>
  );
}
