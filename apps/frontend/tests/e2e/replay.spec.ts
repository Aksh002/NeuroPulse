import {expect,test} from '@playwright/test';

test('real FD001 replay warms up, predicts, pauses, resumes and resets', async ({page}) => {
  await page.goto('/');
  await expect(page.getByRole('heading',{name:'Degradation observatory'})).toBeVisible();
  await expect(page.getByLabel('Engine')).not.toHaveValue('');
  await expect(page.getByLabel('RUL model')).not.toHaveValue('');
  await page.getByLabel('Replay speed').selectOption('10');
  await page.getByRole('button',{name:'Start replay'}).click();
  const cycle=page.locator('.card').filter({hasText:'Operating cycle'}).locator('strong');
  await expect.poll(async()=>Number(await cycle.textContent()),{timeout:60000}).toBeGreaterThanOrEqual(30);
  const rul=page.locator('.card').filter({hasText:'Predicted RUL'}).locator('strong');
  await expect(rul).not.toHaveText('Unavailable');
  await page.getByRole('button',{name:'Pause'}).click();
  const paused=Number(await cycle.textContent());
  await page.waitForTimeout(600);
  await expect(cycle).toHaveText(String(paused));
  await page.getByRole('checkbox',{name:/Show actual RUL/}).check();
  await page.getByRole('button',{name:'Resume replay'}).click();
  await expect.poll(async()=>Number(await cycle.textContent())).toBeGreaterThan(paused);
  await page.getByRole('button',{name:'Pause'}).click();
  await page.getByRole('button',{name:'Reset'}).click();
  await expect(cycle).toHaveText('0');
  await expect(rul).toHaveText('Unavailable');
  await page.getByRole('button',{name:'Resume replay'}).focus();
  await page.keyboard.press('Enter');
  await expect.poll(async()=>Number(await cycle.textContent())).toBeGreaterThan(0);
  await page.getByRole('button',{name:'Pause'}).focus();
  await page.keyboard.press('Space');
  await page.getByRole('link',{name:'Models'}).focus();
  await page.keyboard.press('Enter');
  await expect(page.getByRole('heading',{name:'Neural model comparison'})).toBeVisible();
  await page.getByRole('link',{name:'Experiments'}).click();
  await expect(page.getByRole('heading',{name:'Experiment results'})).toBeVisible();
  await expect(page.getByRole('table')).toBeVisible();
});

test('an unavailable API produces an explicit error', async ({page}) => {
  await page.route('**/api/v1/**',route=>route.fulfill({
    status:503,contentType:'application/json',body:JSON.stringify({error:{message:'API unavailable'}})
  }));
  await page.goto('/');
  await expect(page.locator('.notice.error')).toContainText('API unavailable');
});
