'use strict';
const $ = (id) => document.getElementById(id);
const escapeHTML = (value) => String(value).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
const COLORS = ['#2256d9', '#d77819', '#a244ba', '#14815e'];
const BASE = ['mage', 'visits', 'gained', 'sex_bin', 'habit_bin', 'whitemom_bin'];
const NUMERIC = ['fage','mage','visits','gained'];
const PRESETS = {
  base: {label: 'Base · modelo 2', features: BASE, terms: []},
  gained: {label: 'Base + gained²', features: BASE, terms: [{kind:'square',a:'gained'}]},
  visits: {label: 'Base + visits²', features: BASE, terms: [{kind:'square',a:'visits'}]},
  interactions: {label: 'Base + interacciones', features: BASE, terms: [{kind:'interaction',a:'sex_bin',b:'gained'},{kind:'interaction',a:'habit_bin',b:'gained'}]},
  all: {label: 'Todas las originales', features: ['fage',...BASE,'mature_bin','marital_bin'], terms: []},
  median: {label: 'Sin variables · mediana', features: [], terms: []}
};
let meta, models = [], activeId = 1, nextId = 4, axis = 'gained', view = 'curve';
let revision = 0, timer, aborter, busy = true;
const copy = x => JSON.parse(JSON.stringify(x));
const current = () => models.find(m => m.id === activeId);
const label = f => f.replace('_bin','');
const termLabel = t => t.kind === 'square' ? `${label(t.a)}²` : t.kind === 'interaction' ? `${label(t.a)} × ${label(t.b)}` : `1(${label(t.a)} ≥ ${t.value})`;
const fmt = x => Number(x).toFixed(4).replace('.', ',');
const configOf = m => ({features: m.features, terms: m.terms, alpha: m.alpha});
const options = (values, selected, format = x => x) => values.map(v => `<option value="${escapeHTML(v)}" ${v===selected?'selected':''}>${escapeHTML(format(v))}</option>`).join('');
function newModel(id, color, name, preset) {return {id,color,name,visible:true,alpha:0,...copy(PRESETS[preset]),result:null};}

function renderModels() {
  $('models').innerHTML = models.map(m => `<div class="model ${m.id===activeId?'active':''}" style="--model-color:${m.color}">
    <input class="model-visibility" type="checkbox" data-visible="${m.id}" aria-label="Mostrar ${escapeHTML(m.name)} en gráficos" ${m.visible?'checked':''}>
    <button class="model-choice" data-select="${m.id}" aria-pressed="${m.id===activeId}"><span class="model-name">${escapeHTML(m.name)}</span><span class="model-meta">${m.features.length+m.terms.length} términos · ${m.alpha?'Ridge':'Lineal'}${!m.features.length?' / mediana':''}</span></button>
    <span class="model-score" title="MAE de validación">${m.result?fmt(m.result.mae):'…'}</span>
    ${models.length>1?`<button class="remove-model" data-remove="${m.id}" aria-label="Eliminar ${escapeHTML(m.name)}">×</button>`:''}</div>`).join('');
  $('add-model').disabled = models.length >= 4;
}

function renderEditor() {
  const m = current();
  $('editor').style.setProperty('--model-color',m.color);
  const fieldNames = meta.features.map(f => f.id);
  const vars = meta.features.map(f => `<label class="variable" title="${escapeHTML(f.label)} · ${escapeHTML(f.unit)}"><input type="checkbox" data-feature="${f.id}" ${m.features.includes(f.id)?'checked':''}><span class="var-code">${label(f.id)}</span><span class="var-desc">${escapeHTML(f.label)}</span></label>`).join('');
  const terms = m.terms.map((t,i) => {
    const f = meta.features.find(f=>f.id===t.a);
    return `<div class="term"><div class="term-head"><span>${escapeHTML(termLabel(t))}</span><button data-remove-term="${i}" aria-label="Quitar ${escapeHTML(termLabel(t))}">×</button></div>${t.kind==='threshold'?`<div class="threshold-edit"><input type="range" min="${f.min}" max="${f.max}" step="1" value="${t.value}" data-threshold="${i}" aria-label="Umbral de ${label(t.a)}"><input type="number" min="${f.min}" max="${f.max}" step="1" value="${t.value}" data-threshold-number="${i}" aria-label="Valor del umbral"></div>`:''}</div>`;
  }).join('');
  $('editor').innerHTML = `<div class="editor-head"><label for="model-name">Editando</label><input id="model-name" class="name-input" maxlength="40" value="${escapeHTML(m.name)}" aria-label="Nombre del modelo"></div>
    <label class="field">Partir de <select id="preset"><option value="">Elegir variante…</option>${Object.entries(PRESETS).map(([key,p])=>`<option value="${key}">${p.label}</option>`).join('')}</select></label>
    <h3>Variables de entrenamiento</h3><div class="variables-grid">${vars}</div>
    <h3>Transformaciones</h3>${terms||'<p class="small-note">Sin términos adicionales.</p>'}
    <details class="term-builder"><summary>+ Agregar transformación</summary><div class="builder-controls">
    <select id="term-kind" class="wide" aria-label="Tipo de transformación"><option value="square">Cuadrado · x²</option><option value="interaction">Interacción · x × z</option><option value="threshold">Umbral · 1(x ≥ valor)</option></select>
    <select id="term-a" aria-label="Primera variable">${options(NUMERIC,'gained',label)}</select>
    <select id="term-b" aria-label="Segunda variable" hidden>${options(fieldNames,'sex_bin',label)}</select>
    <input id="term-value" type="number" value="12" aria-label="Valor del nuevo umbral" hidden>
    </div><button id="add-term" class="builder-add">Agregar al modelo</button></details>
    <p class="small-note">Se conservan las variables principales. Quitar una variable quita también sus transformaciones.</p>
    <h3>Ajuste</h3><label class="field">Método <select id="method"><option value="0" ${m.alpha===0?'selected':''}>Regresión lineal</option><option value="ridge" ${m.alpha>0?'selected':''}>Ridge</option></select></label>
    ${m.alpha>0?`<label class="field">Intensidad α <select id="alpha">${options([0.01,0.1,1,10,100,1000],m.alpha,x=>String(x))}</select></label><p class="small-note">Las variables se estandarizan dentro de cada fold.</p>`:''}`;
}

function showError(message) {$('error').textContent=message; $('error').hidden=false;}
function clearError() {$('error').hidden=true;}
function schedule() {
  revision++;
  busy=true;
  $('status').textContent='Recalculando cinco folds…';
  $('plot').setAttribute('aria-busy','true');
  clearTimeout(timer);
  timer=setTimeout(evaluateModels,180);
}

async function evaluateModels() {
  const myRevision=revision;
  if(aborter) aborter.abort();
  aborter=new AbortController();
  try {
    const snapshot=models.map(m=>({id:m.id,config:configOf(m)}));
    const response=await fetch('/api/evaluate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({models:snapshot.map(m=>m.config),axis}),signal:aborter.signal});
    const data=await response.json();
    if(!response.ok) throw new Error(data.error||'No se pudo actualizar.');
    if(myRevision!==revision) return;
    snapshot.forEach((s,i)=>{const m=models.find(m=>m.id===s.id); if(m)m.result=data.results[i];});
    busy=false;
    clearError();
    renderModels();
    draw();
    $('status').textContent='Actualizado · cinco folds fijos';
    $('plot').setAttribute('aria-busy','false');
  } catch(error) {
    if(error.name==='AbortError'||myRevision!==revision)return;
    busy=false;
    $('status').textContent='No se pudo actualizar';
    $('plot').setAttribute('aria-busy','false');
    showError(error.message+' Si cerraste la app, volvé a abrirla con «Abrir app.cmd».');
  }
}

function plotLayout() {
  const feature=meta.features.find(f=>f.id===axis);
  const xtitle=view==='prediction'?'Peso real (lb)':`${feature.label} (${feature.unit})`;
  const ytitle=view==='residual'?'Real − predicho (lb)':view==='prediction'?'Peso predicho (lb)':'Peso al nacer (lb)';
  return {margin:{l:65,r:25,t:25,b:58},paper_bgcolor:'#ffffff',plot_bgcolor:'#ffffff',
    font:{family:'Segoe UI, Arial, sans-serif',size:13,color:'#647184'},hovermode:'closest',dragmode:'pan',
    uirevision:axis+'-'+view,showlegend:true,legend:{orientation:'h',x:0,y:1.08,font:{size:12}},
    xaxis:{title:{text:xtitle,standoff:14},gridcolor:'#e2e8f1',zerolinecolor:'#a5b5cd',showline:true,linecolor:'#b8c4d6',ticks:'outside',minor:{showgrid:true,gridcolor:'#f3f5f9'},automargin:true},
    yaxis:{title:{text:ytitle,standoff:14},gridcolor:'#e2e8f1',zerolinecolor:'#a5b5cd',showline:true,linecolor:'#b8c4d6',ticks:'outside',minor:{showgrid:true,gridcolor:'#f3f5f9'},automargin:true}};
}

function draw() {
  const visible=models.filter(m=>m.visible&&m.result);
  const traces=[];
  const x=meta.data[axis];
  const rowData=meta.rows.map((r,i)=>[r,meta.y[i]]);
  if(view==='curve') {
    traces.push({type:'scatter',mode:'markers',x,y:meta.y,name:'Pesos observados',marker:{color:'#8c99ac',size:5,opacity:.35},customdata:rowData,hovertemplate:'Fila %{customdata[0]}<br>X: %{x}<br>Peso real: %{y:.2f} lb<extra></extra>'});
    visible.forEach(m=>traces.push({type:'scatter',mode:'lines',x:m.result.curveX,y:m.result.curveY,name:m.name,line:{color:m.color,width:3},hovertemplate:'X: %{x:.2f}<br>Peso del perfil: %{y:.3f} lb<extra>%{fullData.name}</extra>'}));
    const profile=models.find(m=>m.result)?.result.profile;
    $('plot-title').textContent='Curvas de los modelos';
    $('plot-note').textContent='Curvas: ajuste con los 540 registros; se varía solo el eje X. Resto del perfil: '+(profile?meta.features.filter(f=>f.id!==axis&&!(axis==='mage'&&f.id==='mature_bin')).map(f=>`${label(f.id)}=${profile[f.id]}`).join(' · '):'')+'. Puntos grises: pesos reales. Estas curvas no calculan el MAE.';
  } else {
    if(view==='prediction') {
      const values=[...meta.y,...visible.flatMap(m=>m.result.pred)];
      const lo=Math.min(...values)-.2,hi=Math.max(...values)+.2;
      traces.push({type:'scatter',mode:'lines',x:[lo,hi],y:[lo,hi],name:'Predicción perfecta',line:{color:'#8b98ac',dash:'dash',width:1.5},hoverinfo:'skip'});
    } else {
      traces.push({type:'scatter',mode:'lines',x:[Math.min(...x),Math.max(...x)],y:[0,0],line:{color:'#8b98ac',dash:'dash',width:1.5},name:'Error cero',hoverinfo:'skip'});
    }
    visible.forEach((m,k)=>traces.push({type:'scatter',mode:'markers',x:view==='prediction'?meta.y:x,y:view==='prediction'?m.result.pred:meta.y.map((y,i)=>y-m.result.pred[i]),name:m.name,marker:{color:m.color,size:6,opacity:.58,symbol:['circle','diamond','square','cross'][COLORS.indexOf(m.color)]},customdata:meta.rows.map((r,i)=>[r,m.result.foldIds[i],meta.y[i],m.result.pred[i]]),hovertemplate:'Fila %{customdata[0]} · Fold %{customdata[1]}<br>Real: %{customdata[2]:.3f} lb<br>Predicho: %{customdata[3]:.3f} lb<br>Valor Y: %{y:.3f}<extra>%{fullData.name}</extra>'}));
    $('plot-title').textContent=view==='prediction'?'Predicciones sobre casos no vistos':'Errores sobre casos no vistos';
    $('plot-note').textContent=view==='prediction'?'Cada punto es una predicción de validación. Cuanto más cerca de la diagonal, menor es el error. Los símbolos distinguen los modelos.':'Residuo = peso real − predicho. Valores positivos: el modelo subestima el peso; valores negativos: lo sobreestima. Buscá patrones o curvaturas.';
  }
  if(!visible.length)$('plot-note').textContent='Activá el círculo de al menos un modelo para mostrarlo en los gráficos.';
  const config={responsive:true,displaylogo:false,scrollZoom:true,modeBarButtonsToRemove:['lasso2d','select2d'],toImageButtonOptions:{filename:'TP1-modelos',format:'png',scale:2}};
  Plotly.react('plot',traces,plotLayout(),config);
  const folds=visible.map(m=>({type:'bar',x:['1','2','3','4','5'],y:m.result.folds,name:m.name,marker:{color:m.color},hovertemplate:'Fold %{x}<br>MAE: %{y:.4f} lb<extra>%{fullData.name}</extra>'}));
  Plotly.react('foldplot',folds,{margin:{l:40,r:8,t:3,b:30},barmode:'group',bargap:.27,paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',showlegend:false,font:{family:'Segoe UI, sans-serif',size:12,color:'#718098'},xaxis:{title:'Fold',type:'category',fixedrange:true},yaxis:{rangemode:'tozero',gridcolor:'#e2e7ef',fixedrange:true}}, {responsive:true,displayModeBar:false});
  renderScores();
}

function renderScores() {
  const ready=models.filter(m=>m.result),reference=ready[0]?.result.mae;
  const max=Math.max(...ready.map(m=>m.result.mae),.001);
  $('scores').innerHTML=ready.map((m,i)=>{const delta=m.result.mae-reference;return `<div class="score-row ${!m.visible?'muted':''}" title="R²: ${fmt(m.result.r2)} · Desvío entre folds: ${fmt(m.result.std)}"><div class="score-name"><span class="dot" style="background:${m.color}"></span><span>${escapeHTML(m.name)}</span></div><div class="score-value">${fmt(m.result.mae)}</div><div class="delta ${delta<-.00005?'better':''}">${i===0?'ref.':(delta>=0?'+':'')+fmt(delta)}</div><div class="score-bar"><i style="background:${m.color};width:${m.result.mae/max*100}%"></i></div></div>`;}).join('');
  $('equations').innerHTML=ready.map(m=>{
    const number=x=>Number(x).toPrecision(4);
    let formula=`${number(m.result.intercept)}`;
    m.result.coefficients.forEach(c=>{formula+=` ${c.value>=0?'+':'−'} ${number(Math.abs(c.value))} · ${c.term}`;});
    return `<div class="equation"><b style="color:${m.color}">${escapeHTML(m.name)}</b> · R² CV ${fmt(m.result.r2)} · MAE entrenamiento ${fmt(m.result.trainMae)}<br><code>peso = ${escapeHTML(formula)}</code>${!m.features.length?'<br>Referencia trivial: mediana del entrenamiento de cada fold.':''}</div>`;
  }).join('');
}

$('models').addEventListener('click',e=>{
  const select=e.target.closest('[data-select]'),remove=e.target.closest('[data-remove]');
  if(select){activeId=Number(select.dataset.select);renderModels();renderEditor();}
  if(remove&&models.length>1){const id=Number(remove.dataset.remove);models=models.filter(m=>m.id!==id);if(activeId===id)activeId=models[0].id;renderModels();renderEditor();schedule();}
});
$('models').addEventListener('change',e=>{if(e.target.matches('[data-visible]')){models.find(m=>m.id===Number(e.target.dataset.visible)).visible=e.target.checked;draw();}});
$('add-model').addEventListener('click',()=>{
  if(models.length>=4)return;
  const source=current(),color=COLORS.find(c=>!models.some(m=>m.color===c));
  const m={...copy(source),id:nextId++,color,name:'Modelo '+String.fromCharCode(65+COLORS.indexOf(color)),result:null,visible:true};
  models.push(m);activeId=m.id;renderModels();renderEditor();schedule();
});
$('editor').addEventListener('change',e=>{
  const m=current(),el=e.target;
  if(el.matches('[data-feature]')){
    const f=el.dataset.feature;
    m.features=el.checked?[...m.features,f]:m.features.filter(x=>x!==f);
    m.terms=m.terms.filter(t=>m.features.includes(t.a)&&(t.kind!=='interaction'||m.features.includes(t.b)));
    renderModels();renderEditor();schedule();
  } else if(el.id==='preset'&&el.value){const p=copy(PRESETS[el.value]);m.features=p.features;m.terms=p.terms;m.alpha=0;renderModels();renderEditor();schedule();}
  else if(el.id==='method'){m.alpha=el.value==='ridge'?10:0;renderModels();renderEditor();schedule();}
  else if(el.id==='alpha'){m.alpha=Number(el.value);schedule();}
  else if(el.id==='term-kind'){
    const k=el.value;$('term-a').innerHTML=options(k==='interaction'?meta.features.map(f=>f.id):NUMERIC,'gained',label);
    $('term-b').hidden=k!=='interaction';$('term-value').hidden=k!=='threshold';
  } else if(el.matches('[data-threshold-number]')){
    const i=Number(el.dataset.thresholdNumber),f=meta.features.find(f=>f.id===m.terms[i].a),val=Number(el.value);
    if(!Number.isFinite(val)||val<f.min||val>f.max){showError(`Elegí un umbral entre ${f.min} y ${f.max}.`);return;}
    m.terms[i].value=val;renderEditor();schedule();
  }
});
$('editor').addEventListener('input',e=>{
  const m=current(),el=e.target;
  if(el.id==='model-name'){m.name=el.value.trim()||'Modelo';renderModels();if(!busy)draw();}
  if(el.matches('[data-threshold]')){
    const i=Number(el.dataset.threshold);m.terms[i].value=Number(el.value);
    el.closest('.term').querySelector('.term-head span').textContent=termLabel(m.terms[i]);
    el.closest('.term').querySelector('input[type=number]').value=el.value;schedule();
  }
});
$('editor').addEventListener('click',e=>{
  const m=current(),remove=e.target.closest('[data-remove-term]');
  if(remove){m.terms.splice(Number(remove.dataset.removeTerm),1);renderModels();renderEditor();schedule();}
  if(e.target.id==='add-term'){
    if(m.terms.length>=12){showError('Usá hasta 12 transformaciones por modelo.');return;}
    const kind=$('term-kind').value,a=$('term-a').value,t={kind,a};
    if(kind==='interaction'){t.b=$('term-b').value;if(t.a===t.b){showError('Para multiplicar una variable por sí misma, elegí «Cuadrado».');return;}}
    if(kind==='threshold'){t.value=Number($('term-value').value);const f=meta.features.find(f=>f.id===a);if(!Number.isFinite(t.value)||t.value<f.min||t.value>f.max){showError(`Elegí un umbral entre ${f.min} y ${f.max}.`);return;}}
    if(m.terms.some(x=>JSON.stringify(x)===JSON.stringify(t))){showError('Esa transformación ya está en el modelo.');return;}
    m.terms.push(t);if(!m.features.includes(a))m.features.push(a);if(t.b&&!m.features.includes(t.b))m.features.push(t.b);
    renderModels();renderEditor();schedule();
  }
});
$('axis').addEventListener('change',e=>{axis=e.target.value;schedule();});
document.querySelectorAll('[data-view]').forEach(button=>button.addEventListener('click',()=>{
  view=button.dataset.view;document.querySelectorAll('[data-view]').forEach(b=>{b.classList.toggle('selected',b===button);b.setAttribute('aria-pressed',String(b===button));});
  $('axis').disabled=view==='prediction';if(models.some(m=>m.result))draw();
}));
$('help-button').addEventListener('click',()=>{$('help').hidden=!$('help').hidden;$('help-button').setAttribute('aria-expanded',String(!$('help').hidden));window.dispatchEvent(new Event('resize'));});

async function init(){
  try{
    const response=await fetch('/api/meta');if(!response.ok)throw new Error('No se pudieron leer los datos.');meta=await response.json();
    $('dataset').textContent=`${meta.n} registros · ${meta.folds} folds · libras`;
    $('axis').innerHTML=options(NUMERIC,axis,f=>`${label(f)} · ${meta.features.find(x=>x.id===f).unit}`);
    models=[newModel(1,COLORS[0],'A · Base','base'),newModel(2,COLORS[1],'B · gained²','gained'),newModel(3,COLORS[2],'C · Interacciones','interactions')];
    renderModels();renderEditor();await evaluateModels();
  }catch(error){$('status').textContent='No se pudo iniciar';showError(error.message);}
}
init();
