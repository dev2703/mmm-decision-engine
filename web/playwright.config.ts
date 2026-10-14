import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  fullyParallel: true,
  timeout: 30000,
  use: { baseURL: "http://127.0.0.1:3000", trace: "retain-on-failure" },
  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        channel: process.env.PLAYWRIGHT_CHANNEL || undefined,
      },
    },
  ],
  webServer: [
    {
      command: "npm run dev",
      url: "http://127.0.0.1:3000",
      reuseExistingServer: !process.env.CI,
      timeout: 60000,
    },
    {
      command: "npm run start -- --port 3101",
      url: "http://127.0.0.1:3101",
      timeout: 60000,
      env: {
        APP_ENV: "local",
        DASHBOARD_USERNAME: "test-reader",
        DASHBOARD_PASSWORD: "test-only-password-for-browser-check",
      },
    },
  ],
});
