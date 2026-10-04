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
      await page.screenshot({path:`ui-artifacts/home-${width}-${theme}.png`,fullPage:true});
      const axe = await new AxeBuilder({page}).withTags(['wcag2a','wcag2aa','wcag21aa']).analyze();
      assert.deepEqual(axe.violations.map(v=>({id:v.id,nodes:v.nodes.map(n=>n.target)})),[],`${width}px ${theme}: accessibility`);
    }
    // A legal real draw can have zero hits. Mock only the browser response to
    // exercise positive-match and tiny-probability presentation deterministically.
    const snap=await page.evaluate(()=>JSON.parse(document.getElementById('initial-data').textContent));
    const mega=snap.products.find(p=>p.product==='mega645'), r=mega.latest;
    mega.comparisons=[{status:'matched',product:mega.product,target_id:r.draw_id,target_date:r.draw_date.slice(0,10),made_at:r.draw_date,engine:'ml',registered:true,result:r,tickets:[{numbers:r.numbers,matched_numbers:r.numbers,hits:6,bonus_hit:false,special:null,tier:'jackpot1'}]}];
    mega.next_forecast={target_id:r.draw_id+1,based_on_id:r.draw_id,target_date:r.draw_date.slice(0,10),target_time:r.draw_date,made_at:r.draw_date,engine:'ml',registered:false,note:'Browser test fixture',components:[{name:'main',top:[{numbers:r.numbers,p_model:1e-9,p_fair:1e-9}]}]};
    snap.generated_at=new Date(Date.parse(snap.generated_at)+60000).toISOString();
    await page.route('**/data/dashboard.json?*',route=>route.fulfill({status:200,contentType:'application/json',body:JSON.stringify(snap)}));
    await page.locator('#refresh').click();
    await page.waitForFunction(()=>!document.getElementById('refresh').disabled);
    const matchAxe=await new AxeBuilder({page}).include('#comparisons').withTags(['wcag2a','wcag2aa','wcag21aa']).analyze();
    assert.deepEqual(matchAxe.violations.map(v=>v.id),[],`${width}px: accessible match highlights`);
    await page.unroute('**/data/dashboard.json?*');
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
    assert(await detail.getAttribute('open')!==null,'opened catalogue survives draw selection');
    snap.generated_at=new Date(Date.parse(snap.generated_at)+60000).toISOString();
    for (const kind of ['products','warnings','timestamp']) {
      const bad=JSON.parse(JSON.stringify(snap));
      bad.generated_at=new Date(Date.parse(snap.generated_at)+60000).toISOString();
      if(kind==='products') bad.products[0].draws=null;
      if(kind==='warnings') delete bad.warnings;
      if(kind==='timestamp') bad.generated_at='2099-01-01T00:00:00Z';
      await page.route('**/data/dashboard.json?*',route=>route.fulfill({status:200,contentType:'application/json',body:JSON.stringify(bad)}));
      await page.locator('#refresh').click();
      await page.waitForFunction(()=>!document.getElementById('refresh').disabled);
      assert.equal(await page.locator('#draw-keno').inputValue(),selectedId,'malformed refresh retains last good DOM');
      assert((await page.locator('#update-status').textContent()).includes('Chưa lấy được'),'malformed refresh is rejected');
      await page.unroute('**/data/dashboard.json?*');
    }
    snap.generated_at=new Date(Date.parse(snap.generated_at)+60000).toISOString();
    await page.route('**/data/dashboard.json?*',route=>route.fulfill({status:200,contentType:'application/json',body:JSON.stringify(snap)}));
    await page.locator('#refresh').click();
    await page.waitForFunction(()=>!document.getElementById('refresh').disabled);
    assert(!(await page.locator('#update-status').textContent()).includes('Chưa lấy được'),'valid refresh recovers after malformed payload');
    await page.unroute('**/data/dashboard.json?*');
    snap.generated_at=new Date(Date.parse(snap.generated_at)+60000).toISOString();
    await page.route('**/data/dashboard.json?*',route=>route.fulfill({status:200,contentType:'application/json',body:JSON.stringify(snap)}));
    await detail.locator('summary').focus();
    await page.evaluate(()=>document.dispatchEvent(new Event('visibilitychange')));
    await page.waitForFunction(()=>!document.getElementById('refresh').disabled);
    assert(await page.locator('.card-details summary').evaluate(element=>element===document.activeElement),'catalogue summary retains keyboard focus');
    snap.generated_at=new Date(Date.parse(snap.generated_at)+60000).toISOString();
    await page.locator('.result-card .source-link').focus();
    await page.evaluate(()=>document.dispatchEvent(new Event('visibilitychange')));
    await page.waitForFunction(()=>!document.getElementById('refresh').disabled);
    assert(await page.locator('.result-card .source-link').evaluate(element=>element===document.activeElement),'source link retains keyboard focus');
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
