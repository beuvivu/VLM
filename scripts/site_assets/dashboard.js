'use strict';
(() => {
  const initial = document.getElementById('initial-data');
  if (!initial) return;
  let snapshot = JSON.parse(initial.textContent);
  let selected = new URLSearchParams(location.search).get('product') || 'all';
  const draws = new Map(), expanded = new Set();
  let busy = false;
  const $ = id => document.getElementById(id);
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const number = value => new Intl.NumberFormat('vi-VN').format(value);
  const percent = value => new Intl.NumberFormat('vi-VN',{maximumSignificantDigits:4}).format(value*100)+'%';
  const money = value => value == null ? 'Chưa cập nhật' : `${number(value)} đ`;
  const time = (value, dateOnly = false) => value ? new Intl.DateTimeFormat('vi-VN', {timeZone:'Asia/Ho_Chi_Minh', day:'2-digit', month:'2-digit', year:'numeric', ...(dateOnly ? {} : {hour:'2-digit',minute:'2-digit'})}).format(new Date(value)) : 'Chưa xác minh giờ quay';
  const did = value => `#${String(value).padStart(5,'0')}`;
  const short = {mega645:'Mega 6/45',power655:'Power 6/55',lotto535:'Lotto 5/35',max3d:'Max 3D / 3D+',max3dpro:'Max 3D Pro',keno:'Keno',bingo18:'Bingo18'};
  const icons = {mega645:'645',power655:'655',lotto535:'535',max3d:'3D',max3dpro:'Pro',keno:'K',bingo18:'18'};
  const tiers = {jackpot1:'Jackpot',jackpot2:'Jackpot 2',first:'Giải Nhất',second:'Giải Nhì',third:'Giải Ba',fourth:'Giải Tư',fifth:'Giải Năm',consolation:'Khuyến khích'};
  const statusLabels = {matched:'Đã đối chiếu',pending:'Chờ kết quả',date_mismatch:'Cần kiểm tra ngày'};
  const safeUrl = value => { try { const u = new URL(value); return ['http:','https:'].includes(u.protocol) ? u.href : '#'; } catch { return '#'; } };
  const visible = (data=snapshot, choice=selected) => data.products.filter(p => choice === 'all' || p.product === choice);
  function balls(nums, hits = [], {digit = false, bonus = false, keno = false, position = null} = {}) {
    return `<div class="balls${keno?' keno':''}">${nums.map((n,i) => {
      const hit = position ? position[i] : hits.includes(n);
      return `<span class="ball${digit?' digit':''}${bonus?' bonus':''}${hit?' hit':''}"${hit?' role="img" aria-label="'+esc(n)+' · trùng kết quả"':''}>${esc(digit ? n : String(n).padStart(2,'0'))}</span>`;
    }).join('')}</div>`;
  }
  const gameHeader = p => `<div class="game-label"><span class="game-icon${['keno','bingo18'].includes(p.product)?' fast':''}">${icons[p.product]}</span><div><h3>${esc(short[p.product])}</h3><p class="game-schedule">${esc(p.schedule)}</p></div></div>`;
  function resultNumbers(p, r, compact = false) {
    if (p.product.startsWith('max')) return `<div class="max-results">${r.prizes.map(t => `<div class="max-tier"><span>${esc(t.label)}</span>${balls(t.numbers,[],{digit:true})}</div>`).join('')}</div>`;
    return `<div class="balls-line">${balls(r.numbers, [], {keno:p.product==='keno'})}${r.bonus!=null?`<div class="facts"><span class="fact">${p.product==='power655'?'Số phụ':'Số đặc biệt'}</span>${balls([r.bonus],[],{bonus:true})}</div>`:''}</div>${compact?'':factRow(p,r)}`;
  }
  function factRow(p,r) {
    const f = r.facts;
    if (p.product==='keno') return `<div class="facts"><span class="fact">Lớn <strong>${f.large}</strong> / Nhỏ <strong>${f.small}</strong></span><span class="fact">Chẵn <strong>${f.even}</strong> / Lẻ <strong>${f.odd}</strong></span>${f.large===f.small?'<span class="fact">Hòa Lớn–Nhỏ</span>':''}${f.even===f.odd?'<span class="fact">Hòa Chẵn–Lẻ</span>':''}</div>`;
    if (p.product==='bingo18') return `<div class="facts"><span class="fact">Tổng <strong>${f.sum}</strong> · ${esc(f.size)}</span>${Object.entries(f.multiplicity).map(([n,count])=>`<span class="fact">Số ${esc(n)}: <strong>${count} lần</strong></span>`).join('')}</div>`;
    return '';
  }
  function resultCard(p) {
    const r = p.draws.find(r=>r.draw_id === draws.get(p.product)) || p.latest;
    if (!r) return `<article class="result-card" data-product="${p.product}"><div class="card-header">${gameHeader(p)}</div><div class="empty-state"><h3>Chưa có kết quả</h3><p>Hệ thống sẽ hiển thị khi có dữ liệu đã xác thực.</p></div></article>`;
    const matrix = ['mega645','power655','lotto535'].includes(p.product);
    return `<article class="result-card" data-product="${p.product}"><div class="card-header">${gameHeader(p)}<span class="pill teal">KẾT QUẢ</span></div><div class="card-body"><div class="draw-meta"><label class="sr-only" for="draw-${p.product}">Chọn kỳ ${esc(short[p.product])}</label><select class="draw-select" id="draw-${p.product}" data-product="${p.product}">${p.draws.map(d=>`<option value="${d.draw_id}"${d.draw_id===r.draw_id?' selected':''}>${did(d.draw_id)} · ${time(d.draw_date,true)}</option>`).join('')}</select><span>${r.time_precision==='day'?'Ngày quay':'Giờ VN'}<br>${time(r.draw_date,r.time_precision==='day')}</span></div>${resultNumbers(p,r)}${matrix?`<table class="prize-table"><thead><tr><th scope="col">Hạng giải</th><th scope="col" class="amount">${p.product==='lotto535'?'Giá trị / Quỹ':'Giá trị / Quỹ'}</th><th scope="col">Lượt trúng</th></tr></thead><tbody>${r.prizes.map(t=>`<tr><td>${esc(t.label)}${t.pool?' <span class="muted">(quỹ)</span>':''}</td><td class="amount">${money(t.value_vnd)}${t.pool && t.value_vnd!=null && t.source && t.source!==r.source?`<span class="prize-source">Nguồn: ${esc(t.source)}</span>`:""}</td><td>${t.winners==null?'—':number(t.winners)}</td></tr>`).join('')}</tbody></table><p class="scope-note">“—” là chưa có dữ liệu. Jackpot là tổng quỹ của đúng kỳ này.${r.prize_source && r.prize_source!==r.source?` Nguồn bảng giải bổ sung: ${esc(r.prize_source)}.`:""}</p>`:''}</div><details class="card-details" data-detail="prizes-${p.product}"${expanded.has(`prizes-${p.product}`)?' open':''}><summary id="prize-summary-${p.product}">Cơ cấu giải ${p.product==='max3d'?'Max 3D & Max 3D+':'& cách trúng'}</summary><p class="scope-note">Mức thưởng theo lượt chơi 10.000 đ, trước thuế. Jackpot và giải có trần được chia theo quy tắc sản phẩm.</p><table class="catalogue-table"><thead><tr><th scope="col">Giải / Cửa chơi</th><th scope="col">Điều kiện</th><th scope="col">Mức thưởng</th></tr></thead><tbody>${p.prize_catalogue.map(t=>`<tr><td>${t.product?esc(t.product)+'<br>':''}${esc(t.label)}</td><td>${esc(t.condition)}${t.note?`<p class="catalogue-note">${esc(t.note)}</p>`:''}</td><td>${t.value_text?esc(t.value_text)+' đ':t.value_vnd==null?'Chia theo quỹ':money(t.value_vnd)}</td></tr>`).join('')}</tbody></table></details><div class="card-footer"><span>Nguồn: ${esc(r.source)}${r.time_precision==='day'?' · chỉ có ngày':''}</span><a class="source-link" id="source-${p.product}" href="${esc(safeUrl(r.official_url))}" target="_blank" rel="noopener noreferrer">Kết quả Vietlott ↗</a></div></article>`;
  }
  function componentRows(p, f) {
    return f.components.filter(c=>c.name!=='special').map(c=> {
      const exact = c.ranking_exact===false ? 'Thứ hạng gần đúng trong ngân sách tìm kiếm.' : '';
      const special = f.components.find(x=>x.name==='special')?.top?.[0]?.numbers;
      return `${p.product==='keno'?'<p class="scope-note">Dãy tham chiếu 20 số của mô hình; vé Keno chỉ chọn bậc 1–10.</p>':''}${p.product==='max3dpro'?'<p class="scope-note">Gợi ý từng số 3 chữ số; vé Pro cần một cặp hai số.</p>':''}<div class="prediction-list">${c.top.slice(0,5).map((row,i)=>{
        const isMax = p.product.startsWith('max');
        const nums = isMax ? [row.numbers.join('')] : row.numbers;
        const probability = row.p_model!=null ? `P mô hình ${percent(row.p_model)}${row.p_fair!=null?` · ngẫu nhiên ${percent(row.p_fair)}`:''}${isMax?' / mỗi số trong nhóm quay':''}` : 'Bộ số đã lưu của mô hình thống kê';
        return `<div class="prediction-row"><div class="prediction-row-top"><span class="rank">${i+1}</span>${balls(nums,[],{digit:isMax,keno:p.product==='keno'})}</div>${special?`<div class="probability">Số đặc biệt tham chiếu: <strong>${esc(special[0])}</strong></div>`:''}<p class="probability">${esc(probability)}</p></div>`;
      }).join('')}</div>${exact?`<p class="scope-note">${exact}</p>`:''}`;
    }).join('');
  }
  function predictionCard(p) {
    const f = p.next_forecast;
    return `<article class="prediction-card" data-product="${p.product}"><div class="card-header">${gameHeader(p)}<span class="pill ${f?.registered?'teal':'amber'}">${f?.registered?'ĐÃ ĐĂNG KÝ':'THAM KHẢO'}</span></div>${f?`<div class="card-body"><div class="draw-meta"><strong>Kỳ ${did(f.target_id)}</strong><span>${time(f.target_time)}</span></div>${componentRows(p,f)}</div><div class="prediction-meta"><p>${esc(f.note)}</p><p>${f.engine==='ml'?'Ensemble ML':'Mô hình thống kê'} · dữ liệu tới ${did(f.based_on_id)} · ${time(f.made_at)}</p></div>`:`<div class="empty-state"><h3>Chờ dự báo kỳ kế tiếp</h3><p>Chưa có bộ số đồng bộ với kỳ kết quả mới nhất. Dự báo sẽ xuất hiện sau khi mô hình cập nhật.</p><a class="text-link" id="history-${p.product}" href="forecast.html#${p.product}">Xem lịch sử phân tích ↗</a></div>`}</article>`;
  }
  function comparisonCard(p,c) {
    const pending = c.status==='pending';
    const header = `<div class="comparison-top"><div><h3>${esc(short[p.product])} <span class="muted">/ ${did(c.target_id)}</span></h3><p class="meta">Đã lưu ${time(c.made_at)} · kỳ ${time(c.target_date+'T00:00:00+07:00',true)}</p></div><span class="pill ${c.status==='matched'?'teal':'amber'}">${statusLabels[c.status]||esc(c.status)}</span></div>`;
    if (pending) return `<article class="comparison-card">${header}<div class="empty-state"><p>Đã qua giờ nhận dự báo. Chưa có kết quả của đúng kỳ này để đối chiếu.</p></div></article>`;
    if (c.status!=='matched') return `<article class="comparison-card">${header}<div class="empty-state"><p>Ngày kết quả khác ngày dự báo đã đăng ký. Tạm dừng chấm, cần kiểm tra dữ liệu.</p></div></article>`;
    return `<article class="comparison-card">${header}<div class="comparison-content"><div class="comparison-result"><p class="eyebrow">KẾT QUẢ ĐÃ NHẬN</p>${resultNumbers(p,c.result,true)}<p class="scope-note">Nguồn: ${esc(c.result.source)} · ${time(c.result.draw_date,c.result.time_precision==='day')}</p></div><div class="comparison-rows"><p class="eyebrow">BỘ SỐ ĐÃ LƯU</p>${c.tickets.map((t,i)=>{
      const isMax = p.product.startsWith('max'), isBingo = p.product==='bingo18';
      const nums = isMax ? [t.symbol] : t.numbers;
      const matched = isMax && t.hits>0 ? [t.symbol] : t.matched_numbers || [];
      let caption = isMax ? (t.hits ? `Trùng ${t.hits} lần · ${t.tiers.join(', ')}` : 'Không trùng số đầy đủ') : isBingo ? `${t.position_hits}/3 đúng vị trí · ${t.multiset_hits}/3 trùng đa tập${t.sum_match?' · đúng tổng':''}${t.exact?' · đúng cả bộ':''}` : `${t.hits}/${t.numbers.length} số chính${t.bonus_hit?' + số phụ/đặc biệt':''}${t.tier?' · '+(tiers[t.tier]||t.tier):' · chưa đạt hạng giải'}`;
      if (p.product==='keno') caption = `${t.hits}/20 số trong dãy tham chiếu`;
      return `<div class="prediction-row"><div class="prediction-row-top"><span class="rank">${i+1}</span>${balls(nums,matched,{digit:isMax,keno:p.product==='keno',position:isBingo?t.position_matches:null})}</div>${t.special!=null?`<p class="scope-note">Số đặc biệt đã lưu: ${t.special}</p>`:''}<p class="match-caption">${esc(caption)}</p></div>`;
    }).join('')}</div></div><div class="comparison-bottom">Bản ${c.engine==='ml'?'ML':'thống kê'} đã đăng ký trước kỳ quay · đối chiếu tự động theo mã kỳ · hạng giải minh họa theo bộ số, không xác nhận tiền thưởng thực nhận.</div></article>`;
  }
  function heroMarkup(data) {
    const products = data.products.filter(p=>p.latest);
    const withPool = products.filter(p=>p.latest.prizes.some(t=>t.pool && t.value_vnd!=null));
    const p = withPool.find(p=>p.product==='mega645') || withPool[0] || products[0];
    if (!p) return '<p class="muted">Chờ kết quả đầu tiên</p>';
    const r = p.latest, jackpot = r.prizes.find(t=>t.pool && t.value_vnd!=null);
    return `<div class="hero-panel-top"><span class="pill">${jackpot?'JACKPOT MỚI NHẤT':'THEO DÕI KỲ QUAY'}</span><span class="orbit-icon" aria-hidden="true">✦</span></div><p class="hero-game">${esc(short[p.product])} · kỳ ${did(r.draw_id)}</p><p class="jackpot-amount">${jackpot?`${number(jackpot.value_vnd)} <small>đ</small>`:'Kết quả đã về.'}</p><p class="scope-note">${jackpot?`${esc(jackpot.label)} · quỹ giải của kỳ ${time(r.draw_date,true)}`:'Bảng Jackpot của kỳ mới đang chờ cập nhật.'}</p>${p.product.startsWith('max')?balls(r.numbers.slice(0,2),[],{digit:true}):balls(r.numbers.slice(0,6))}<div class="hero-panel-bottom"><span>Nguồn: ${esc(r.source)}</span><a id="hero-results" href="#results">Xem đầy đủ các giải ↗</a></div>`;
  }
  function rememberDetails() {
    // toggle events are queued; read the live state before replacing any cards.
    document.querySelectorAll('.card-details').forEach(detail=>{
      const key=detail.dataset.detail;
      detail.open?expanded.add(key):expanded.delete(key);
    });
  }
  function renderContent() {
    rememberDetails();
    const current = visible();
    $('results-grid').classList.toggle('filtered',selected!=='all');
    $('predictions-grid').classList.toggle('filtered',selected!=='all');
    $('results-grid').innerHTML=current.map(resultCard).join('');
    $('predictions-grid').innerHTML=current.map(predictionCard).join('');
    renderComparisons();
  }
  function comparisonsMarkup(data=snapshot, choice=selected) {
    const filter = $('comparison-status').value;
    const rows = visible(data,choice).flatMap(p=>p.comparisons.map(c=>({p,c}))).filter(({c})=>filter==='all'||c.status===filter).sort((a,b)=>a.c.target_date.localeCompare(b.c.target_date)*-1 || b.c.target_id-a.c.target_id);
    return rows.length?rows.map(({p,c})=>comparisonCard(p,c)).join(''):'<div class="empty-state"><h3>Chưa có kỳ phù hợp để đối chiếu</h3><p>Dự báo đã đăng ký sẽ được so sánh tự động khi kết quả của đúng kỳ quay về. Dự báo tham khảo không được tính vào lịch sử kiểm chứng.</p></div>';
  }
  function renderComparisons() { $('comparisons-list').innerHTML=comparisonsMarkup(); }
  function renderSnapshot(next=snapshot) {
    rememberDetails();
    const choice=next.products.some(p=>p.product===selected)?selected:'all';
    // Prepare every section before replacing the last good data or any live DOM.
    const current=visible(next,choice);
    const markup={
      'hero-panel':heroMarkup(next),
      stats:[['◈',`${next.stats.results}/7`,'Sản phẩm có kết quả'],['↗',next.stats.registered_next,'Dự báo kỳ tới đã đăng ký'],['✓',next.stats.compared_draws,'Kỳ đã tự động đối chiếu']].map(([icon,value,label])=>`<div class="stat"><span class="stat-icon" aria-hidden="true">${icon}</span><div><strong>${value}</strong><p>${label}</p></div></div>`).join(''),
      'product-filters':[['all','Tất cả'],...next.products.map(p=>[p.product,short[p.product]])].map(([code,label])=>`<button class="filter-button" id="filter-${code}" data-product="${code}" type="button" aria-pressed="${choice===code}">${esc(label)}</button>`).join(''),
      'results-grid':current.map(resultCard).join(''),
      'predictions-grid':current.map(predictionCard).join(''),
      'comparisons-list':comparisonsMarkup(next,choice)
    };
    const focusId=document.activeElement?.id;
    for(const [id,html] of Object.entries(markup)) $(id).innerHTML=html;
    snapshot=next; selected=choice;
    for(const id of ['results-grid','predictions-grid']) $(id).classList.toggle('filtered',selected!=='all');
    if(focusId) $(focusId)?.focus({preventScroll:true});
    updateStatus();
    $('data-warning').hidden=!next.warnings.length;
    $('data-warning').textContent=next.warnings.length?`Dữ liệu cần kiểm tra: ${next.warnings.join(' · ')}`:'';
  }
  function validateSnapshot(data) {
    const obj=v=>v!==null && typeof v==='object' && !Array.isArray(v);
    const str=v=>typeof v==='string';
    const int=v=>Number.isSafeInteger(v) && v>=0;
    const stamp=v=>str(v) && Number.isFinite(Date.parse(v));
    const date=v=>str(v) && /^\d{4}-\d{2}-\d{2}$/.test(v) && stamp(v);
    const array=(v,test)=>Array.isArray(v) && v.every(test);
    const nullable=(v,test)=>v==null || test(v);
    const moneyValue=v=>Number.isFinite(v) && v>=0;
    const probability=v=>moneyValue(v) && v<=1;
    const numbers=v=>array(v,n=>int(n)||str(n)&&/^\d{3}$/.test(n));
    const catalogue=t=>obj(t) && str(t.label) && str(t.condition) && nullable(t.value_vnd,moneyValue) && nullable(t.note,str) && nullable(t.product,str) && nullable(t.value_text,str);
    const result=(r,code)=>obj(r) && int(r.draw_id) && stamp(r.draw_date) && numbers(r.numbers) && nullable(r.bonus,int) && ['day','second'].includes(r.time_precision) && str(r.source) && str(r.official_url) && array(r.prizes,t=>obj(t)&&str(t.label)&&(code.startsWith('max')?numbers(t.numbers):nullable(t.value_vnd,moneyValue)&&nullable(t.winners,int))) && obj(r.facts) && (code!=='keno'||['large','small','even','odd'].every(k=>int(r.facts[k]))) && (code!=='bingo18'||int(r.facts.sum)&&str(r.facts.size)&&obj(r.facts.multiplicity)&&Object.values(r.facts.multiplicity).every(int));
    const forecast=f=>obj(f) && int(f.target_id) && int(f.based_on_id) && stamp(f.made_at) && nullable(f.target_time,stamp) && nullable(f.target_date,date) && str(f.engine) && typeof f.registered==='boolean' && str(f.note) && array(f.components,c=>obj(c)&&str(c.name)&&array(c.top,t=>obj(t)&&numbers(t.numbers)&&nullable(t.p_model,probability)&&nullable(t.p_fair,probability)));
    const ticket=(t,code)=>obj(t) && numbers(t.numbers) && (code.startsWith('max')?str(t.symbol)&&int(t.hits)&&array(t.tiers,str):code==='bingo18'?int(t.position_hits)&&int(t.multiset_hits)&&array(t.position_matches,b=>typeof b==='boolean')&&typeof t.sum_match==='boolean'&&typeof t.exact==='boolean':int(t.hits)&&array(t.matched_numbers,int)&&typeof t.bonus_hit==='boolean'&&nullable(t.special,int)&&nullable(t.tier,str));
    const comparison=(c,code)=>obj(c) && int(c.target_id) && date(c.target_date) && stamp(c.made_at) && str(c.engine) && ['pending','matched','date_mismatch'].includes(c.status) && (c.status!=='matched'||result(c.result,code)&&array(c.tickets,t=>ticket(t,code)));
    if(!obj(data)||data.schema_version!==1||!stamp(data.generated_at)||Date.parse(data.generated_at)>Date.now()+300000||!obj(data.stats)||!['products','results','registered_next','compared_draws'].every(k=>int(data.stats[k]))||!array(data.warnings,str)||!Array.isArray(data.products)||data.products.length!==7) throw new Error('schema');
    const codes=new Set();
    for(const p of data.products) {
      if(!obj(p)||!Object.hasOwn(short,p.product)||codes.has(p.product)||!str(p.schedule)||!nullable(p.latest,r=>result(r,p.product))||!array(p.draws,r=>result(r,p.product))||!array(p.prize_catalogue,catalogue)||!nullable(p.next_forecast,forecast)||!array(p.comparisons,c=>comparison(c,p.product))) throw new Error('schema');
      codes.add(p.product);
    }
  }
  function updateStatus(message) {
    const age=(Date.now()-new Date(snapshot.generated_at).getTime())/3600000;
    $('update-status').textContent=message || `Cập nhật ${time(snapshot.generated_at)} (giờ VN) · ${age>1?'bản dữ liệu đã hơn 1 giờ':'tự kiểm tra bản mới mỗi phút'}`;
  }
  $('product-filters').addEventListener('click',event=>{
    const button=event.target.closest('[data-product]'); if (!button) return;
    selected=button.dataset.product;
    document.querySelectorAll('.filter-button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.product===selected)));
    const url=new URL(location); selected==='all'?url.searchParams.delete('product'):url.searchParams.set('product',selected); history.replaceState(null,'',url);
    renderContent();
  });
  $('results-grid').addEventListener('change',event=>{
    if (!event.target.matches('.draw-select')) return;
    rememberDetails();
    draws.set(event.target.dataset.product,Number(event.target.value));
    const focus=event.target.id;
    $('results-grid').innerHTML=visible().map(resultCard).join('');
    $(focus)?.focus({preventScroll:true});
  });
  $('comparison-status').addEventListener('change',renderComparisons);
  async function refresh(manual=false) {
    if (busy || (!manual && document.hidden)) return;
    busy=true; $('refresh').disabled=true;
    try {
      const response=await fetch(`data/dashboard.json?t=${Date.now()}`,{cache:'no-store',signal:AbortSignal.timeout(15000)});
      if (!response.ok) throw new Error('http');
      const next=await response.json();
      validateSnapshot(next);
      if (new Date(next.generated_at)>=new Date(snapshot.generated_at)) {
        if (next.generated_at!==snapshot.generated_at) renderSnapshot(next);
        else updateStatus(manual?`Đã kiểm tra · bản mới nhất ${time(snapshot.generated_at)} (giờ VN)`:undefined);
      }
    } catch { updateStatus(`Chưa lấy được bản mới · giữ dữ liệu ${time(snapshot.generated_at)} (giờ VN). Sẽ tự thử lại.`); }
    finally { busy=false; $('refresh').disabled=false; }
  }
  $('refresh').addEventListener('click',()=>refresh(true));
  document.addEventListener('visibilitychange',()=>{if(!document.hidden)refresh();});
  renderSnapshot();
  refresh();
  setInterval(refresh,60000);
})();
