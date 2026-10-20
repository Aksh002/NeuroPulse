import {defineConfig} from '@playwright/test';
import {resolve} from 'node:path';

export default defineConfig({
  testDir: './tests/e2e',
  timeout: 90000,
  retries: 0,
  workers: 1,
  webServer: [
    {
      command: 'uv run --no-sync uvicorn apps.backend.main:app --host 127.0.0.1 --port 8000 --workers 1',
      cwd: resolve(process.cwd(),'../..'),
      url: 'http://127.0.0.1:8000/api/v1/health/ready',
      reuseExistingServer: !process.env.CI,
      timeout: 120000,
    },
    {
      command: 'npm run dev',
      cwd: process.cwd(),
      url: 'http://localhost:3000',
      reuseExistingServer: !process.env.CI,
      timeout: 120000,
    },
  ],
  use: {
    baseURL: 'http://localhost:3000',
    browserName: 'chromium',
    launchOptions: {
      executablePath: process.env.NEUROPULSE_BROWSER_PATH ||
        'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
      headless: true,
    },
  },
});
