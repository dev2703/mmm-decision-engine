import { test, expect, type Page } from "@playwright/test";

// Explicit HTTP contract fixtures, never shipped as dashboard/demo results.
const projectId = "11111111-1111-4111-8111-111111111111";
const modelId = "22222222-2222-4222-8222-222222222222";
const datasetId = "33333333-3333-4333-8333-333333333333";
const secondId = "44444444-4444-4444-8444-444444444444";
const model = {
  id: modelId,
  dataset_id: datasetId,
  status: "SUCCEEDED",
  configuration: { channels: ["search_spend", "meta_spend"], chains: 4 },
  training_window: { rows: 100 },
};
async function fixture(page: Page, release = "BLOCK") {
  await page.route("**/api/backend/**", async (route) => {
    const url = new URL(route.request().url());
    const path = url.pathname.replace("/api/backend", "");
    let data: unknown = {};
    if (path === "/capabilities")
      data = { read_only: false, agent_enabled: false };
    else if (path === "/projects")
      data = [
        {
          id: projectId,
          name: "Contract fixture",
          current_dataset_id: datasetId,
        },
      ];
    else if (path.endsWith("/model-runs"))
      data = [model, { ...model, id: secondId }];
    else if (path.endsWith("/datasets"))
      data = [
        {
          id: datasetId,
          quality_status: "WARNING",
          date_start: "2022-01-03",
          date_end: "2024-12-23",
          hash: "fixture",
        },
      ];
    else if (path.endsWith("/health"))
      data = {
        decision_status: release,
        diagnostics: {},
        policy: { reasons: [], channel_movement_limits: {} },
      };
    else if (path.endsWith("/model-comparison") || path.endsWith("/runs"))
      data = { items: [], total: 0, next_offset: null };
    else if (path.endsWith("/predictions"))
      data = { rows: [], provenance: { limitations: [] } };
    await route.fulfill({ json: data });
  });
}

test("blocked evidence cannot queue a scenario; agent remains disconnected", async ({
  page,
}) => {
  await fixture(page);
  await page.goto("/decision-lab");
  await expect(
    page.getByRole("button", { name: "Validate constraints" }),
  ).toBeDisabled();
  await expect(
    page.getByText("The selected model is blocked.", { exact: false }),
  ).toBeVisible();
  await page
    .getByRole("link", { name: "Ask DecisionGuard", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "Ask DecisionGuard" }),
  ).toBeDisabled();
  await expect(
    page.getByText("Conversational exploration is not connected"),
  ).toBeVisible();
});

test("constraints require a fresh review after editing and on model switch", async ({
  page,
}) => {
  await fixture(page, "WARN");
  let submitted: unknown;
  await page.route("**/constraints/validate", async (route) => {
    submitted = route.request().postDataJSON();
    await route.fulfill({
      json: {
        valid: true,
        effective_bounds: { search_spend: [90, 110], meta_spend: [80, 120] },
        release: { state: "WARN" },
        evaluation_run_id: "fixture",
      },
    });
  });
  await page.goto("/decision-lab");
  await page.getByLabel("Total weekly budget", { exact: true }).fill("200");
  await page.getByLabel("Search current", { exact: true }).fill("100");
  await page.getByLabel("Meta current", { exact: true }).fill("100");
  await page.getByLabel("Search movement", { exact: true }).fill("10");
  await page.getByRole("button", { name: "Validate constraints" }).click();
  await expect(
    page.getByRole("button", { name: "Create and queue scenario" }),
  ).toBeVisible();
  expect(submitted).toMatchObject({
    total: 200,
    current: { search_spend: 100, meta_spend: 100 },
    max_movement: { search_spend: 0.1 },
  });
  await page.getByLabel("Total weekly budget", { exact: true }).fill("210");
  await expect(
    page.getByRole("button", { name: "Create and queue scenario" }),
  ).toHaveCount(0);
  await page
    .getByLabel("Selected model", { exact: true })
    .selectOption(secondId);
  await expect(
    page.getByLabel("Total weekly budget", { exact: true }),
  ).toHaveValue("");
});

test("a late old-model response never replaces current evidence", async ({
  page,
}) => {
  await fixture(page);
  let releaseOld: () => void = () => {};
  const oldReady = new Promise<void>((resolve) => {
    releaseOld = resolve;
  });
  await page.route(`**/model-runs/${modelId}/health`, async (route) => {
    await oldReady;
    await route.fulfill({
      json: {
        decision_status: "PASS",
        diagnostics: {},
        policy: { reasons: [], channel_movement_limits: {} },
      },
    });
  });
  await page.goto("/model-health");
  await expect(page.getByLabel("Selected model", { exact: true })).toHaveValue(
    modelId,
  );
  await page
    .getByLabel("Selected model", { exact: true })
    .selectOption(secondId);
  await expect(
    page.getByText("Budget recommendations are blocked"),
  ).toBeVisible();
  releaseOld();
  await expect(page.getByText("Current release: PASS")).toHaveCount(0);
});

test("unavailable backend evidence is explicit", async ({ page }) => {
  await page.route("**/api/backend/**", (route) =>
    route.fulfill({ status: 502, json: { detail: "Backend unavailable" } }),
  );
  await page.goto("/");
  await expect(page.locator("main [role=alert]")).toContainText(
    "Backend unavailable",
  );
  await expect(page.getByText("Create sample workspace")).toHaveCount(0);
});

test("proxy rejects foreign origins, unexpected paths and oversized writes", async ({
  request,
}) => {
  expect((await request.get("/api/backend/openapi.json")).status()).toBe(404);
  expect(
    (
      await request.post("/api/backend/projects", {
        data: { name: "No write" },
        headers: { Origin: "https://foreign.invalid" },
      })
    ).status(),
  ).toBe(403);
  expect(
    (
      await request.post("/api/backend/projects", {
        data: { name: "x".repeat(40000) },
        headers: { Origin: "http://127.0.0.1:3000" },
      })
    ).status(),
  ).toBe(413);
});

test("dashboard access also protects the API proxy", async ({ request }) => {
  for (const path of ["/", "/api/backend/projects"]) {
    const response = await request.get(`http://127.0.0.1:3101${path}`);
    expect(response.status()).toBe(401);
  }
  const valid = await request.get("http://127.0.0.1:3101/", {
    headers: {
      Authorization: `Basic ${Buffer.from("test-reader:test-only-password-for-browser-check").toString("base64")}`,
    },
  });
  expect(valid.status()).toBe(200);
  expect(await valid.text()).not.toContain(
    "test-only-password-for-browser-check",
  );
});

test("real seeded evidence renders all seven pages on desktop and mobile", async ({
  page,
  request,
}) => {
  test.skip(
    process.env.DEMO_E2E !== "1",
    "Requires the documented seeded PostgreSQL demo and native API",
  );
  const projects = await (await request.get("/api/backend/projects")).json();
  expect(projects.length).toBeGreaterThan(0);
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await expect(
    page.getByRole("img", {
      name: "Actual and predicted weekly revenue with 90% posterior interval, AUD",
    }),
  ).toBeVisible();
  await expect(
    page.locator(".stats-grid").first().getByText("BLOCK", { exact: true }),
  ).toBeVisible();
  await expect(
    page.locator(".stat").nth(1).locator(".stat-value"),
  ).not.toHaveText("—");
  await page.screenshot({
    path: "test-results/overview-desktop.png",
    fullPage: true,
  });
  for (const path of [
    "/data-quality",
    "/model-lab",
    "/model-health",
    "/decision-lab",
    "/ask",
    "/runs",
  ]) {
    await page.goto(path);
    await expect(page.locator("h1")).toBeVisible();
    await expect(page.locator(".loading")).toHaveCount(0);
    await expect(page.locator("main [role=alert]")).toHaveCount(0);
  }
  await page.setViewportSize({ width: 390, height: 844 });
  for (const path of [
    "/",
    "/data-quality",
    "/model-lab",
    "/model-health",
    "/decision-lab",
    "/ask",
    "/runs",
  ]) {
    await page.goto(path);
    await expect(page.locator("h1")).toBeVisible();
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBeTruthy();
  }
  await page.getByRole("button", { name: "Open navigation" }).click();
  await page.getByRole("link", { name: "Overview", exact: true }).click();
  await expect(page.locator(".sidebar.open")).toHaveCount(0);
  await expect(page.locator(".chart")).toBeVisible();
  await expect(
    page.locator(".stats-grid").first().getByText("BLOCK", { exact: true }),
  ).toBeVisible();
  await expect(
    page.locator(".stat").nth(1).locator(".stat-value"),
  ).not.toHaveText("—");
  await page.screenshot({
    path: "test-results/overview-mobile.png",
    fullPage: true,
  });
  expect(errors).toEqual([]);
});

test("completed historical scenario shows uncertainty without recommendation authority", async ({
  page,
}) => {
  await fixture(page, "WARN");
  const selected = {
    allocation: { search_spend: 100, meta_spend: 100 },
    expected_modeled_media_revenue: 1000,
    media_revenue_interval_90: [800, 1200],
    probability_beat_current: 0.5,
    downside_probability: 0.5,
    stability_p95: 0.1,
    expected_improvement: 0,
    extrapolated_channels: [],
  };
  await page.route("**/constraints/validate", (route) =>
    route.fulfill({
      json: {
        valid: true,
        effective_bounds: { search_spend: [90, 110], meta_spend: [80, 120] },
        release: { state: "WARN" },
        evaluation_run_id: "fixture",
      },
    }),
  );
  await page.route(`**/model-runs/${modelId}/scenarios`, (route) =>
    route.fulfill({ json: { id: datasetId } }),
  );
  await page.route(`**/scenarios/${datasetId}/optimize`, (route) =>
    route.fulfill({
      json: {
        id: datasetId,
        status: "SUCCEEDED",
        evidence_current: false,
        recommendation_allowed: false,
        summary: {
          selected,
          alternatives: { conservative: selected },
          baseline_allocation: selected.allocation,
          horizon_weeks: 13,
        },
      },
    }),
  );
  await page.goto("/decision-lab");
  await page.getByLabel("Total weekly budget", { exact: true }).fill("200");
  await page.getByLabel("Search current", { exact: true }).fill("100");
  await page.getByLabel("Meta current", { exact: true }).fill("100");
  await page.getByRole("button", { name: "Validate constraints" }).click();
  await page.getByRole("button", { name: "Create and queue scenario" }).click();
  await expect(
    page.getByText("This result does not authorize a recommendation"),
  ).toBeVisible();
  await expect(
    page.getByText("90% modeled media revenue interval:", { exact: false }),
  ).toContainText("$800–$1,200");
  await expect(
    page.getByRole("img", {
      name: "Current versus proposed constant weekly budget by channel",
    }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Start another scenario" }).click();
  await expect(page.getByText("Scenario outcome")).toHaveCount(0);
});
