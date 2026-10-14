"use client";
import { useState } from "react";
import Link from "next/link";
import { ArrowUpRight, MessageSquare, Send, ShieldCheck } from "lucide-react";
import { useWorkspace } from "@/components/workspace";
import {
  Badge,
  ErrorNotice,
  EvidenceDetails,
  Notice,
  PageTitle,
  Panel,
} from "@/components/ui";
import { api } from "@/lib/api";
import type { AgentAnswer } from "@/lib/types";

export default function Ask() {
  const { project, model } = useWorkspace();
  return <Conversation key={`${project?.id}:${model?.id}`} />;
}
function Conversation() {
  const { project, model, agentEnabled } = useWorkspace();
  const [message, setMessage] = useState("");
  const [response, setResponse] = useState<AgentAnswer>();
  const [error, setError] = useState<Error>();
  const [busy, setBusy] = useState(false);
  async function submit() {
    if (!project || !agentEnabled) return;
    setBusy(true);
    setError(undefined);
    setResponse(undefined);
    try {
      setResponse(
        await api<AgentAnswer>("/agent/query", {
          method: "POST",
          body: JSON.stringify({
            project_id: project.id,
            message: `${message}${model ? `\nSelected model run: ${model.id}` : ""}`,
          }),
        }),
      );
    } catch (failure) {
      setError(failure as Error);
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <PageTitle
        eyebrow="05 / ASK DECISIONGUARD"
        title="Ask better questions of your data."
        description="Explore the evidence conversationally, with the same decision gates as the rest of the workspace."
      />
      {!agentEnabled && (
        <Notice title="Conversational exploration is not connected">
          Connect a chat provider to ask questions. You can inspect all model,
          validation and scenario evidence directly in the workspace today.
        </Notice>
      )}
      <div className="ask-intro">
        <span className="ask-icon">
          <MessageSquare size={26} />
        </span>
        <h2>
          Start with the decision.
          <br />
          <span>Follow the evidence.</span>
        </h2>
        <p>
          Every numerical answer must come from a verified tool. The assistant
          cannot override a blocked model.
        </p>
      </div>
      <div className="question-grid">
        {[
          [
            "Why is this model restricted?",
            "/model-health",
            "Inspect validation and release reasons",
          ],
          [
            "How uncertain is Meta’s ROI?",
            "/model-lab",
            "Explore posterior channel evidence",
          ],
          [
            "Can we change the allocation?",
            "/decision-lab",
            "Validate constraints and compare risks",
          ],
        ].map(([question, href, description]) => (
          <Link href={href} key={href} className="question-card">
            <strong>
              {question}
              <ArrowUpRight size={16} />
            </strong>
            <p>{description}</p>
          </Link>
        ))}
      </div>
      <Panel
        title="Evidence conversation"
        subtitle="No live campaign changes. Every answer exposes its sources."
      >
        <form
          className="ask-form"
          onSubmit={(event) => {
            event.preventDefault();
            void submit();
          }}
        >
          <label htmlFor="question">Your question</label>
          <textarea
            id="question"
            value={message}
            onChange={(event) => setMessage(event.target.value)}
            maxLength={7500}
            required
            disabled={!agentEnabled || busy}
            placeholder={
              agentEnabled
                ? "What would you like to understand?"
                : "Connect an agent provider to ask a question."
            }
          />
          <div className="form-actions">
            <span className="muted">
              <ShieldCheck size={14} /> Verified evidence only
            </span>
            <button
              className="button"
              disabled={!agentEnabled || !project || busy}
            >
              <Send size={15} />
              {busy ? "Exploring…" : "Ask DecisionGuard"}
            </button>
          </div>
        </form>
      </Panel>
      <ErrorNotice error={error} />
      {response && (
        <Panel
          title={response.message}
          action={<Badge value={response.status} />}
        >
          {response.status === "error" && (
            <p>
              You can still inspect the evidence directly in{" "}
              <Link href="/model-health">Model Health</Link> and{" "}
              <Link href="/model-lab">Model Lab</Link>.
            </p>
          )}
          {response.evidence.map((item) => (
            <EvidenceDetails
              key={item.id}
              data={item.evidence}
              title={`${item.id} · ${item.tool}`}
            />
          ))}
          <EvidenceDetails
            data={response.trace}
            title="Inspect tool trace and usage"
          />
          {response.issues.map((issue, index) => (
            <Notice key={index} title={issue.tool}>
              {issue.status}
            </Notice>
          ))}
        </Panel>
      )}
    </>
  );
}
