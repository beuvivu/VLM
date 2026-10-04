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
  const visible = () => snapshot.products.filter(p => selected === 'all' || p.product === selected);
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
    return `<article class="result-card" data-product="${p.product}"><div class="card-header">${gameHeader(p)}<span class="pill teal">KẾT QUẢ</span></div><div class="card-body"><div class="draw-meta"><label class="sr-only" for="draw-${p.product}">Chọn kỳ ${esc(short[p.product])}</label><select class="draw-select" id="draw-${p.product}" data-product="${p.product}">${p.draws.map(d=>`<option value="${d.draw_id}"${d.draw_id===r.draw_id?' selected':''}>${did(d.draw_id)} · ${time(d.draw_date,true)}</option>`).join('')}</select><span>${r.time_precision==='day'?'Ngày quay':'Giờ VN'}<br>${time(r.draw_date,r.time_precision==='day')}</span></div>${resultNumbers(p,r)}${matrix?`<table class="prize-table"><thead><tr><th scope="col">Hạng giải</th><th scope="col" class="amount">${p.product==='lotto535'?'Giá trị / Quỹ':'Giá trị / Quỹ'}</th><th scope="col">Lượt trúng</th></tr></thead><tbody>${r.prizes.map(t=>`<tr><td>${esc(t.label)}${t.pool?' <span class="muted">(quỹ)</span>':''}</td><td class="amount">${money(t.value_vnd)}</td><td>${t.winners==null?'—':number(t.winners)}</td></tr>`).join('')}</tbody></table><p class="scope-note">“—” là chưa có dữ liệu. Jackpot là tổng quỹ của đúng kỳ này.</p>`:''}</div><details class="card-details" data-detail="prizes-${p.product}"${expanded.has(`prizes-${p.product}`)?' open':''}><summary>Cơ cấu giải ${p.product==='max3d'?'Max 3D & Max 3D+':'& cách trúng'}</summary><p class="scope-note">Mức thưởng theo lượt chơi 10.000 đ, trước thuế. Jackpot và giải có trần được chia theo quy tắc sản phẩm.</p><table class="catalogue-table"><thead><tr><th scope="col">Giải / Cửa chơi</th><th scope="col">Điều kiện</th><th scope="col">Mức thưởng</th></tr></thead><tbody>${p.prize_catalogue.map(t=>`<tr><td>${t.product?esc(t.product)+'<br>':''}${esc(t.label)}</td><td>${esc(t.condition)}${t.note?`<p class="catalogue-note">${esc(t.note)}</p>`:''}</td><td>${t.value_text?esc(t.value_text)+' đ':t.value_vnd==null?'Chia theo quỹ':money(t.value_vnd)}</td></tr>`).join('')}</tbody></table></details><div class="card-footer"><span>Nguồn: ${esc(r.source)}${r.time_precision==='day'?' · chỉ có ngày':''}</span><a class="source-link" href="${esc(safeUrl(r.official_url))}" target="_blank" rel="noopener noreferrer">Kết quả Vietlott ↗</a></div></article>`;
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
    return `<article class="prediction-card" data-product="${p.product}"><div class="card-header">${gameHeader(p)}<span class="pill ${f?.registered?'teal':'amber'}">${f?.registered?'ĐÃ ĐĂNG KÝ':'THAM KHẢO'}</span></div>${f?`<div class="card-body"><div class="draw-meta"><strong>Kỳ ${did(f.target_id)}</strong><span>${time(f.target_time)}</span></div>${componentRows(p,f)}</div><div class="prediction-meta"><p>${esc(f.note)}</p><p>${f.engine==='ml'?'Ensemble ML':'Mô hình thống kê'} · dữ liệu tới ${did(f.based_on_id)} · ${time(f.made_at)}</p></div>`:`<div class="empty-state"><h3>Chờ dự báo kỳ kế tiếp</h3><p>Chưa có bộ số đồng bộ với kỳ kết quả mới nhất. Dự báo sẽ xuất hiện sau khi mô hình cập nhật.</p><a class="text-link" href="forecast.html#${p.product}">Xem lịch sử phân tích ↗</a></div>`}</article>`;
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
  function renderHero() {
    const products = snapshot.products.filter(p=>p.latest);
    const withPool = products.filter(p=>p.latest.prizes.some(t=>t.pool && t.value_vnd!=null));
    const p = withPool.find(p=>p.product==='mega645') || withPool[0] || products[0];
    if (!p) { $('hero-panel').innerHTML='<p class="muted">Chờ kết quả đầu tiên</p>'; return; }
    const r = p.latest, jackpot = r.prizes.find(t=>t.pool && t.value_vnd!=null);
    $('hero-panel').innerHTML = `<div class="hero-panel-top"><span class="pill">${jackpot?'JACKPOT MỚI NHẤT':'THEO DÕI KỲ QUAY'}</span><span class="orbit-icon" aria-hidden="true">✦</span></div><p class="hero-game">${esc(short[p.product])} · kỳ ${did(r.draw_id)}</p><p class="jackpot-amount">${jackpot?`${number(jackpot.value_vnd)} <small>đ</small>`:'Kết quả đã về.'}</p><p class="scope-note">${jackpot?`${esc(jackpot.label)} · quỹ giải của kỳ ${time(r.draw_date,true)}`:'Bảng Jackpot của kỳ mới đang chờ cập nhật.'}</p>${p.product.startsWith('max')?balls(r.numbers.slice(0,2),[],{digit:true}):balls(r.numbers.slice(0,6))}<div class="hero-panel-bottom"><span>Nguồn: ${esc(r.source)}</span><a href="#results">Xem đầy đủ các giải ↗</a></div>`;
  }
  function renderContent() {
    const current = visible();
    $('results-grid').classList.toggle('filtered',selected!=='all');
    $('predictions-grid').classList.toggle('filtered',selected!=='all');
    $('results-grid').innerHTML=current.map(resultCard).join('');
    $('predictions-grid').innerHTML=current.map(predictionCard).join('');
    renderComparisons();
  }
  function renderComparisons() {
    const filter = $('comparison-status').value;
    const rows = visible().flatMap(p=>p.comparisons.map(c=>({p,c}))).filter(({c})=>filter==='all'||c.status===filter).sort((a,b)=>a.c.target_date.localeCompare(b.c.target_date)*-1 || b.c.target_id-a.c.target_id);
    $('comparisons-list').innerHTML=rows.length?rows.map(({p,c})=>comparisonCard(p,c)).join(''):'<div class="empty-state"><h3>Chưa có kỳ phù hợp để đối chiếu</h3><p>Dự báo đã đăng ký sẽ được so sánh tự động khi kết quả của đúng kỳ quay về. Dự báo tham khảo không được tính vào lịch sử kiểm chứng.</p></div>';
  }
  function renderSnapshot() {
    if (!snapshot.products.some(p=>p.product===selected)) selected='all';
    const focusId=document.activeElement?.id;
    renderHero();
    $('stats').innerHTML=[['◈',`${snapshot.stats.results}/7`,'Sản phẩm có kết quả'],['↗',snapshot.stats.registered_next,'Dự báo kỳ tới đã đăng ký'],['✓',snapshot.stats.compared_draws,'Kỳ đã tự động đối chiếu']].map(([icon,value,label])=>`<div class="stat"><span class="stat-icon" aria-hidden="true">${icon}</span><div><strong>${value}</strong><p>${label}</p></div></div>`).join('');
    $('product-filters').innerHTML=[['all','Tất cả'],...snapshot.products.map(p=>[p.product,short[p.product]])].map(([code,label])=>`<button class="filter-button" id="filter-${code}" data-product="${code}" type="button" aria-pressed="${selected===code}">${esc(label)}</button>`).join('');
    renderContent();
    if (focusId) document.getElementById(focusId)?.focus({preventScroll:true});
    updateStatus();
    $('data-warning').hidden=!snapshot.warnings.length;
    $('data-warning').textContent=snapshot.warnings.length?`Dữ liệu cần kiểm tra: ${snapshot.warnings.join(' · ')}`:'';
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
    draws.set(event.target.dataset.product,Number(event.target.value));
    const focus=event.target.id;
    $('results-grid').innerHTML=visible().map(resultCard).join('');
    $(focus)?.focus({preventScroll:true});
  });
  $('results-grid').addEventListener('toggle',event=>{
    const key=event.target.dataset.detail;
    if (key) event.target.open?expanded.add(key):expanded.delete(key);
  },true);
  $('comparison-status').addEventListener('change',renderComparisons);
  async function refresh(manual=false) {
    if (busy || (!manual && document.hidden)) return;
    busy=true; $('refresh').disabled=true;
    try {
      const response=await fetch(`data/dashboard.json?t=${Date.now()}`,{cache:'no-store',signal:AbortSignal.timeout(15000)});
      if (!response.ok) throw new Error('http');
      const next=await response.json();
      if (next.schema_version!==1 || !Array.isArray(next.products) || !next.stats || !next.generated_at) throw new Error('schema');
      if (new Date(next.generated_at)>=new Date(snapshot.generated_at)) {
        if (next.generated_at!==snapshot.generated_at) { snapshot=next; renderSnapshot(); }
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
