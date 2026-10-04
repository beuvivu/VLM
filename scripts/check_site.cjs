/* Run against the built site; CI installs Playwright and saves screenshots for review. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const { chromium } = require('playwright');
const AxeBuilder = require('@axe-core/playwright').default;
const base = process.env.VLM_SITE_URL || 'http://127.0.0.1:8765';

(async () => {
  fs.mkdirSync('ui-artifacts', {recursive:true});
  const browser = await chromium.launch({headless:true});
  const errors = [];
  for (const width of [1440,375,414]) {
    const context = await browser.newContext({viewport:{width,height:1000}});
    const page = await context.newPage();
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(base, {waitUntil:'networkidle'});
    assert.equal(await page.locator('.result-card').count(),7,'seven active products are visible');
    assert.equal(await page.locator('.prediction-card').count(),7);
    for (const theme of ['light','dark']) {
      await page.evaluate(theme => {localStorage.setItem('vlm-theme',theme);document.documentElement.dataset.theme=theme;},theme);
      assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'no horizontal page overflow');
      const axe = await new AxeBuilder({page}).withTags(['wcag2a','wcag2aa','wcag21aa']).analyze();
      assert.deepEqual(axe.violations.map(v=>({id:v.id,nodes:v.nodes.map(n=>n.target)})),[],`${width}px ${theme}: accessibility`);
      await page.screenshot({path:`ui-artifacts/home-${width}-${theme}.png`,fullPage:true});
    }
    await page.locator('#filter-keno').click();
    assert.equal(await page.locator('.result-card').count(),1);
    assert.equal(await page.locator('.result-card .balls.keno .ball').count(),20);
    const detail=page.locator('.card-details');
    await detail.locator('summary').click();
    assert(await detail.getAttribute('open')!==null);
    assert(await detail.locator('tbody tr').count()>30,'full Keno prize catalogue');
    const selector=page.locator('#draw-keno');
    const options=await selector.locator('option').allTextContents();
    assert(options.length>1);
    await selector.selectOption({index:1});
    const selectedId=await selector.inputValue();
    const snap=await page.evaluate(()=>JSON.parse(document.getElementById('initial-data').textContent));
    snap.generated_at=new Date(Date.parse(snap.generated_at)+60000).toISOString();
    await page.route('**/data/dashboard.json?*',route=>route.fulfill({status:200,contentType:'application/json',body:JSON.stringify(snap)}));
    await page.locator('#refresh').click();
    await page.waitForFunction(()=>!document.getElementById('refresh').disabled);
    assert.equal(await page.locator('#draw-keno').inputValue(),selectedId,'selected draw survives refresh');
    assert(await page.locator('.card-details').getAttribute('open')!==null,'opened catalogue survives refresh');
    assert.equal(await page.locator('#filter-keno').getAttribute('aria-pressed'),'true');
    await page.unroute('**/data/dashboard.json?*');
    await page.route('**/data/dashboard.json?*',route=>route.fulfill({status:503,body:'unavailable'}));
    await page.locator('#refresh').click();
    await page.waitForFunction(()=>!document.getElementById('refresh').disabled);
    assert((await page.locator('#update-status').textContent()).includes('giữ dữ liệu'));
    assert.equal(await page.locator('#draw-keno').inputValue(),selectedId);
    if (width<760) {
      await page.locator('.menu-toggle').click();
      assert.equal(await page.locator('.menu-toggle').getAttribute('aria-expanded'),'true');
      await page.keyboard.press('Escape');
      assert.equal(await page.locator('.menu-toggle').getAttribute('aria-expanded'),'false');
    }
    await page.unroute('**/data/dashboard.json?*');
    await page.locator('#filter-all').click();
    assert(await page.locator('.comparison-card .hit').count()>0,'stored forecast comparisons highlight full matches');
    const probabilities=await page.locator('.probability').allTextContents();
    assert(!probabilities.some(text=>/P mô hình 0%/.test(text)),'small positive probabilities are not rounded to zero');
    await page.locator('.theme-toggle').click();
    const chosenTheme=await page.locator('html').getAttribute('data-theme');
    await page.reload({waitUntil:'networkidle'});
    assert.equal(await page.locator('html').getAttribute('data-theme'),chosenTheme);
    await page.goto(`${base}/forecast.html`,{waitUntil:'networkidle'});
    assert((await page.locator('h1').textContent()).includes('Phân tích'));
    assert.equal(await page.locator('html').getAttribute('data-theme'),chosenTheme);
    assert(await page.locator('a[href="index.html"]').count()>0);
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'analysis page fits viewport');
    await page.screenshot({path:`ui-artifacts/forecast-${width}.png`,fullPage:true});
    await context.close();
  }
  await browser.close();
  assert.deepEqual(errors,[],'no uncaught browser errors');
  console.log('PASS: 3 viewports, light/dark, accessibility, filters, catalogue, draw selection, refresh failure/success, themes, preserved analysis.');
})().catch(error=>{console.error(error);process.exit(1);});
