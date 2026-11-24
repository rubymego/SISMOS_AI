/* SISMO AI PERÚ · interacciones de la interfaz */
(() => {
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

  /* Menú lateral colapsable (móvil) */
  const sidebar = $('#sidebar');
  const toggle = $('#menu-toggle');
  const backdrop = $('#backdrop');
  const setMenu = (open) => {
    sidebar?.classList.toggle('open', open);
    backdrop?.classList.toggle('show', open);
    toggle?.setAttribute('aria-expanded', String(open));
  };
  toggle?.addEventListener('click', () => setMenu(!sidebar.classList.contains('open')));
  backdrop?.addEventListener('click', () => setMenu(false));
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') setMenu(false); });

  /* Contadores animados de los KPI */
  $$('[data-count]').forEach((el) => {
    const target = parseFloat(el.dataset.count);
    const decimals = parseInt(el.dataset.decimals || '0', 10);
    const suffix = el.dataset.suffix || '';
    const fmt = (v) => v.toLocaleString('es-PE', { minimumFractionDigits: decimals, maximumFractionDigits: decimals }) + suffix;
    if (!Number.isFinite(target) || window.matchMedia('(prefers-reduced-motion: reduce)').matches) { el.textContent = fmt(target); return; }
    const start = performance.now();
    const step = (now) => {
      const t = Math.min(1, (now - start) / 1100);
      el.textContent = fmt(target * (1 - Math.pow(1 - t, 3)));
      if (t < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  });

  /* Estado de carga de mapas */
  $$('[data-loading] iframe').forEach((frame) => {
    frame.addEventListener('load', () => frame.parentElement.classList.remove('loading'));
  });

  /* Validación en el navegador (el servidor valida de nuevo) */
  const FIELD_LABELS = { magnitud: 'La magnitud', profundidad_km: 'La profundidad', distancia_km: 'La distancia al epicentro' };
  const validateForm = (form) => {
    for (const input of $$('input[type="number"]', form)) {
      input.removeAttribute('aria-invalid');
      const label = FIELD_LABELS[input.name] || 'El campo';
      const raw = input.value.trim();
      const value = Number(raw.replace(',', '.'));
      let message = '';
      if (!raw) message = `${label} es obligatoria.`;
      else if (!Number.isFinite(value)) message = `${label} debe ser un número (por ejemplo 5.8).`;
      else if (value < Number(input.min) || value > Number(input.max)) message = `${label} debe estar entre ${input.min} y ${input.max}.`;
      if (message) { input.setAttribute('aria-invalid', 'true'); input.focus(); return message; }
    }
    return '';
  };
  const showError = (form, message) => {
    const box = $('.form-error', form);
    if (!box) return;
    box.textContent = message;
    box.hidden = !message;
  };

  const simForm = $('#simulation-form');
  simForm?.addEventListener('submit', (e) => {
    const message = validateForm(simForm);
    showError(simForm, message);
    if (message) { e.preventDefault(); return; }
    const btn = $('button[type="submit"]', simForm);
    btn.disabled = true; btn.classList.add('busy'); $('span', btn).textContent = 'Estimando…';
  });

  /* Simulador rápido del dashboard: consulta /api/estimar (modelo KNN en el servidor) */
  const quick = $('#quick-sim');
  const out = $('#quick-result');
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const renderResult = (r) => {
    const { labels, names, colors } = window.SISMO;
    const rows = labels.map((l) => {
      const p = ((r.probabilities[l] || 0) * 100).toFixed(1);
      return `<div class="prob-row"><span>${names[l]} <small>${l}</small></span><div class="bar"><i style="width:0;background:${colors[l]}" data-w="${p}%"></i></div><b>${p}%</b></div>`;
    }).join('');
    const votes = labels.map((l) => `<span class="vote" style="--c:${colors[l]}">${names[l]} ${r.votes[l]}</span>`).join('');
    const warn = r.fuera_de_rango.length ? `<p class="warn">Atención: ${esc(r.fuera_de_rango.join(', '))} queda fuera del rango de entrenamiento; la estimación es menos fiable.</p>` : '';
    out.innerHTML = `<div class="result-card" style="--accent:${r.color}">
      <div class="result-head"><span class="eyebrow">Intensidad estimada · ${esc(r.departamento)} · K = ${r.k}</span>
      <strong class="result-label">${esc(r.label_name)}</strong><span class="result-range">Mercalli ${esc(r.predicted)}</span></div>
      <div class="prob-list">${rows}</div><p class="votes">Votos de los ${r.k} vecinos: ${votes}</p>${warn}</div>`;
    requestAnimationFrame(() => requestAnimationFrame(() => $$('[data-w]', out).forEach((i) => { i.style.width = i.dataset.w; })));
  };
  quick?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const message = validateForm(quick);
    showError(quick, message);
    if (message) return;
    const btn = $('button[type="submit"]', quick);
    const label = $('span', btn);
    btn.disabled = true; btn.classList.add('busy'); label.textContent = 'Estimando…';
    try {
      const res = await fetch(quick.dataset.api, { method: 'POST', body: new FormData(quick) });
      const data = await res.json();
      if (!res.ok) showError(quick, data.error || 'No se pudo calcular la estimación.');
      else renderResult(data);
    } catch {
      showError(quick, 'No se pudo conectar con el servidor. Verifica que app.py esté en ejecución.');
    } finally {
      btn.disabled = false; btn.classList.remove('busy'); label.textContent = 'Estimar intensidad';
    }
  });

  /* Gráficos interactivos (Chart.js) */
  if (!window.Chart || !window.SISMO) return;
  Chart.defaults.color = '#8fa3c0';
  Chart.defaults.font.family = 'Inter, Segoe UI, sans-serif';
  Chart.defaults.borderColor = 'rgba(42, 66, 104, .5)';
  Chart.defaults.plugins.tooltip.backgroundColor = '#0b1730';
  Chart.defaults.plugins.tooltip.borderColor = '#2a4268';
  Chart.defaults.plugins.tooltip.borderWidth = 1;
  Chart.defaults.plugins.tooltip.padding = 10;
  const S = window.SISMO;
  const make = (id, config) => { const el = document.getElementById(id); if (el) new Chart(el, config); };
  const gridY = { color: 'rgba(42,66,104,.35)' };
  const base = (scales, tooltipLabel) => ({ responsive: true, maintainAspectRatio: false, scales,
    plugins: { legend: { display: false }, tooltip: { callbacks: { label: tooltipLabel } } } });

  if (S.charts) {
    make('chart-magnitudes', { type: 'doughnut', data: { labels: S.charts.magnitudes.labels,
      datasets: [{ data: S.charts.magnitudes.values, backgroundColor: ['#ffb347', '#ff8c32', '#ff3b4e', '#b3122c'], borderColor: '#0e1b31', borderWidth: 3, hoverOffset: 8 }] },
      options: { responsive: true, maintainAspectRatio: false, cutout: '68%',
        plugins: { legend: { position: 'bottom', labels: { boxWidth: 10, usePointStyle: true } },
          tooltip: { callbacks: { label: (c) => ` ${c.label} Mw: ${c.parsed} eventos` } } } } });
    make('chart-departamentos', { type: 'bar', data: { labels: S.charts.departamentos.labels,
      datasets: [{ data: S.charts.departamentos.values, backgroundColor: 'rgba(59,157,255,.85)', hoverBackgroundColor: '#34d1e0', borderRadius: 7, maxBarThickness: 30 }] },
      options: { indexAxis: 'y', ...base({ x: { beginAtZero: true, grid: gridY }, y: { grid: { display: false } } }, (c) => ` ${c.parsed.x} eventos`) } });
  }
  if (S.cv) {
    const minF1 = Math.min(...S.cv.map((s) => s.f1_macro_cv));
    make('chart-k', { type: 'bar', data: { labels: S.cv.map((s) => `K = ${s.k}`),
      datasets: [{ data: S.cv.map((s) => s.f1_macro_cv), borderRadius: 7, maxBarThickness: 46,
        backgroundColor: S.cv.map((s) => (s.k === S.bestK ? '#3b9dff' : 'rgba(59,157,255,.35)')) }] },
      options: base({ x: { grid: { display: false } }, y: { min: Math.max(0, Math.floor((minF1 - 0.03) * 100) / 100), max: 1, grid: gridY } },
        (c) => ` F1 macro: ${c.parsed.y.toFixed(4)} ± ${S.cv[c.dataIndex].f1_std_cv.toFixed(4)}`) });
  }
  if (S.classCounts) {
    const keys = Object.keys(S.classCounts);
    make('chart-classes', { type: 'bar', data: { labels: keys.map((k) => `${S.names[k]} (${k})`),
      datasets: [{ data: keys.map((k) => S.classCounts[k]), backgroundColor: keys.map((k) => S.colors[k]), borderRadius: 7, maxBarThickness: 60 }] },
      options: base({ x: { grid: { display: false } }, y: { beginAtZero: true, grid: gridY } }, (c) => ` ${c.parsed.y} registros`) });
  }
})();
