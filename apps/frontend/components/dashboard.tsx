'use client';
import {useEffect,useRef,useState} from 'react';
import Link from 'next/link';
import {CartesianGrid,Line,LineChart,ResponsiveContainer,Tooltip,XAxis,YAxis} from 'recharts';
import {API,api,formatMetric,Machine,Prediction,Run,Session} from '../lib/api';

type View = 'overview'|'machines'|'replay'|'models'|'experiments';
const nav: View[] = ['overview','machines','replay','models','experiments'];
type Point = {cycle:number;[key:string]:number|null};
export default function Dashboard({view='overview'}:{view?:View}) {
  const [machines,setMachines]=useState<Machine[]>([]),[models,setModels]=useState<Run[]>([]),[runs,setRuns]=useState<Run[]>([]);
  const [machine,setMachine]=useState(''),[model,setModel]=useState(''),[anomaly,setAnomaly]=useState<string|null>(null);
  const [session,setSession]=useState<Session|null>(null),[prediction,setPrediction]=useState<Prediction|null>(null);
  const [sensor,setSensor]=useState('sensor_02'),[sensors,setSensors]=useState<Point[]>([]),[history,setHistory]=useState<Point[]>([]);
  const [speed,setSpeed]=useState(1),[truth,setTruth]=useState(false),[alerts,setAlerts]=useState<{cycle:number;recommendation:string;reason_codes:string[]}[]>([]);
  const [error,setError]=useState(''),[busy,setBusy]=useState(false),[selectedRun,setSelectedRun]=useState<Run|null>(null);
  const sequence=useRef(0), current=useRef<Session|null>(null); current.current=session;

  useEffect(()=>{let active=true;
    Promise.all([api<{items:Machine[]}>('/machines'),api<{items:Run[]}>('/models'),api<{items:Run[]}>('/experiments'),api<{model_version:string;anomaly_model_version:string|null}>('/health/ready')])
      .then(([m,models,r,ready])=>{if(!active)return;setMachines(m.items);setMachine(m.items[0]?.machine_id??'');setModels(models.items);setRuns(r.items);setModel(ready.model_version);setAnomaly(ready.anomaly_model_version);})
      .catch(e=>active&&setError(String(e.message)));
    return()=>{active=false;sequence.current++;};
  },[]);

  async function refresh(active:Session) {
    const token=sequence.current;
    const suffix=`?session_id=${active.session_id}&limit=1000`;
    const [points,predictions,events,evaluation]=await Promise.all([
      api<{items:Point[]}>(`/machines/${machine}/sensors${suffix}`),
      api<{items:Prediction[]}>(`/machines/${machine}/predictions${suffix}`),
      api<{items:typeof alerts}>(`/alerts${suffix}`),
      truth?api<{items:{cycle:number;true_rul_cycles:number}[]}>(`/machines/${machine}/evaluation?session_id=${active.session_id}`):Promise.resolve({items:[]})
    ]);
    if(token!==sequence.current)return;
    if(current.current?.session_id===active.session_id && current.current.generation===active.generation && current.current.cycle>active.cycle)return;
    const actual=new Map(evaluation.items.map(p=>[p.cycle,p.true_rul_cycles]));
    setSensors(points.items);setHistory(predictions.items.map(p=>({cycle:p.cycle,rul:p.rul_cycles,actual:actual.get(p.cycle)??null,anomaly:p.anomaly_score})));
    setAlerts(events.items);setPrediction(active.last_prediction??active.prediction??null);
  }
  useEffect(()=>{if(!session||session.status!=='running')return;let cancelled=false; const token=sequence.current;
    const timer=setTimeout(async()=>{try{
      const next=await api<Session>(`/replays/${session.session_id}/step`,{step_id:crypto.randomUUID(),generation:session.generation});
      if(cancelled||token!==sequence.current)return;current.current=next;setSession(next);await refresh(next);
    }catch(e){if(!cancelled){setError((e as Error).message);setSession(s=>s?{...s,status:'paused'}:s);}}},1000/speed);
    return()=>{cancelled=true;clearTimeout(timer);};
  },[session,speed]); // One request at a time; server verifies generation and serializes steps.
  useEffect(()=>{if(session)refresh(session).catch(e=>setError(e.message));},[truth]);

  async function action(command:'start'|'pause'|'resume'|'reset') {
    setBusy(true);setError('');sequence.current++;
    try {
      let active=session;
      if(!active){active=await api<Session>('/replays',{machine_id:machine,model_version:model,anomaly_model_version:anomaly,speed_cps:speed});}
      const next=await api<Session>(`/replays/${active.session_id}/control`,{action:command,generation:active.generation,speed_cps:speed});
      current.current=next;setSession(next);if(command==='reset'){setSensors([]);setHistory([]);setPrediction(null);setAlerts([]);}else await refresh(next);
    }catch(e){setError((e as Error).message);}finally{setBusy(false);}
  }
  async function changeSelection(value:string,kind:'machine'|'model') {
    sequence.current++; const old=current.current;
    setSession(null);setPrediction(null);setSensors([]);setHistory([]);setAlerts([]);
    if(kind==='machine')setMachine(value);else {setModel(value);setAnomaly(null);}
    if(old)await fetch(`${API}/replays/${old.session_id}`,{method:'DELETE'}).catch(()=>{});
  }
  async function inspect(run:Run){try{setSelectedRun(await api<Run>(`/experiments/${run.run_id}`));}catch(e){setError((e as Error).message);}}
  const latest=session?.last_prediction??prediction;
  const title={overview:'Degradation observatory',machines:'Machine health',replay:'Historical replay',models:'Neural model comparison',experiments:'Experiment results'}[view];
  const chart=(points:Point[],key:string,color='#58e2b4',actual=false)=><div className="chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={points}><CartesianGrid stroke="#243746" strokeDasharray="3 3"/><XAxis dataKey="cycle" stroke="#94a8b8" label={{value:'Operating cycle',position:'insideBottom',offset:-3}}/><YAxis stroke="#94a8b8"/><Tooltip contentStyle={{background:'#132433',borderColor:'#365063'}}/><Line type="monotone" dataKey={key} name={key==='rul'?'Predicted RUL (cycles)':key} stroke={color} dot={false} connectNulls={false} isAnimationActive={false}/>{actual&&<Line dataKey="actual" name="Actual RUL — evaluation only" stroke="#efba79" strokeDasharray="5 4" dot={false} isAnimationActive={false}/>}</LineChart></ResponsiveContainer></div>;
  return <div className="shell"><aside className="sidebar"><div className="brand">Neuro<span>Pulse</span></div><p className="muted">DEGRADATION RESEARCH</p><nav>{nav.map(v=><Link key={v} href={v==='overview'?'/':`/${v}`} className={view===v?'active':''}>{v[0].toUpperCase()+v.slice(1)}</Link>)}</nav><p className="footnote">NASA C-MAPSS · FD001<br/>CPU research prototype</p></aside><main>
    <span className="eyebrow">Neural predictive maintenance</span><h1>{title}</h1><p className="muted">Inspect learned degradation patterns, replay engine history, and compare reproducible experiments.</p><span className="badge">HISTORICAL DATA REPLAY</span>
    <div className="notice">Research demonstration. Recommendations use configurable rules; they are not validated industrial maintenance policies.</div>
    {error&&<div className="notice error" role="alert">{error}</div>}
    {(['overview','machines','replay'] as View[]).includes(view)?<>
      <div className="cards"><div className="card"><span className="muted">Validation engines</span><strong>{machines.length}</strong><small className="muted">Held out from training</small></div><div className="card"><span className="muted">Operating cycle</span><strong>{session?.cycle??0}</strong><small className="muted">Observed replay prefix</small></div><div className="card"><span className="muted">Predicted RUL · cycles</span><strong className="metric">{formatMetric(latest?.rul_cycles)}</strong><small className="muted">{latest?.status==='warming_up'?'Warming up — more history needed':'Neural regression estimate'}</small></div><div className="card"><span className="muted">RUL-derived index</span><strong>{formatMetric(latest?.health_index)}</strong><small className="muted">0–100 display index; not physical health</small></div></div>
      <div className="controls"><label>Engine<select aria-label="Engine" value={machine} onChange={e=>changeSelection(e.target.value,'machine')} disabled={busy||session?.status==='running'}>{machines.map(m=><option key={m.machine_id}>{m.machine_id}</option>)}</select></label><label>RUL model<select aria-label="RUL model" value={model} onChange={e=>changeSelection(e.target.value,'model')} disabled={busy||session?.status==='running'}>{models.filter(m=>m.family!=='autoencoder').map(m=><option key={m.model_version} value={m.model_version}>{m.model_version}</option>)}</select></label><label>Cycles / second<select aria-label="Replay speed" value={speed} onChange={e=>setSpeed(Number(e.target.value))}>{[.5,1,2,5,10].map(s=><option key={s} value={s}>{s}</option>)}</select></label><button className="primary" disabled={busy||!model||!machine||session?.status==='running'||session?.status==='completed'} onClick={()=>action(session?'resume':'start')}>{session?'Resume replay':'Start replay'}</button><button disabled={busy||session?.status!=='running'} onClick={()=>action('pause')}>Pause</button><button disabled={busy||!session} onClick={()=>action('reset')}>Reset</button></div>
      <div className="grid"><div><section className="panel"><h2>Sensor trajectory</h2><label className="muted">Sensor <select aria-label="Sensor" value={sensor} onChange={e=>setSensor(e.target.value)}>{Array.from({length:21},(_,i)=>`sensor_${String(i+1).padStart(2,'0')}`).map(s=><option key={s}>{s}</option>)}</select></label>{sensors.length?chart(sensors,sensor):<div className="empty">Start replay to observe the sensor prefix.</div>}<p className="muted">Raw sensor values; physical units remain unverified.</p></section><section className="panel"><h2>Remaining useful life</h2><label className="muted"><input type="checkbox" checked={truth} onChange={e=>setTruth(e.target.checked)}/> Show actual RUL — evaluation only</label>{history.length?chart(history,'rul','#58e2b4',truth):<div className="empty">Predictions appear after a complete sensor window.</div>}</section><section className="panel"><h2>Reconstruction anomaly signal</h2>{anomaly&&history.length?chart(history,'anomaly','#99a6ff'):<p className="muted">Anomaly branch unavailable for the selected model. Reconstruction error is not a failure probability.</p>}</section></div><aside><section className="panel"><h2>Maintenance assessment</h2><span className="badge">{latest?.recommendation?.replaceAll('_',' ')??'Awaiting prediction'}</span><p className="muted">{latest?.reason_codes?.join(' · ')||'Rules activate when a prediction is available.'}</p><p className="muted">Anomaly threshold: {formatMetric(latest?.anomaly_threshold)}</p><p className="muted">Uncertainty: unavailable</p><p className="muted">The index normalizes RUL against the configured reference and saturates at 100.</p></section><section className="panel"><h2>Alert transitions</h2><ul className="alerts">{alerts.slice(-12).reverse().map((a,i)=><li key={i}>Cycle {a.cycle}<br/><b>{a.recommendation.replaceAll('_',' ')}</b><br/><span className="muted">{a.reason_codes.join(', ')}</span></li>)}</ul>{!alerts.length&&<p className="muted">No alert transitions observed.</p>}</section></aside></div>
    </>:<><section className="panel"><h2>Measured runs · FD001</h2><p className="muted">Uncapped validation truth. Compare matching windows, caps and budgets. Autoencoder values use healthy-proxy reconstruction MSE.</p><div className="table-wrap"><table><thead><tr><th>Architecture</th><th>Seed</th><th>Window</th><th>MAE · cycles</th><th>RMSE · cycles</th><th>NASA score</th><th>Status</th><th>Inspect</th></tr></thead><tbody>{runs.map(r=><tr key={r.run_id}><td>{r.family}</td><td>{r.config.seed}</td><td>{r.config.window_length}</td><td>{formatMetric(r.metrics?.validation?.mae_cycles)}</td><td>{formatMetric(r.metrics?.validation?.rmse_cycles??r.metrics?.healthy_proxy_validation_mse)}</td><td>{formatMetric(r.metrics?.validation?.nasa_score_sum)}</td><td>{r.status}</td><td><button onClick={()=>inspect(r)}>Details</button></td></tr>)}</tbody></table></div>{!runs.length&&<div className="empty">No measured runs available. Run the documented experiment suite.</div>}</section>
    {selectedRun&&<section className="panel"><h2>{selectedRun.family} · seed {selectedRun.config.seed}</h2><p className="muted">{selectedRun.model_version}</p>{selectedRun['history.csv']&&<div className="chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={selectedRun['history.csv']?.map(r=>({epoch:Number(r.epoch),loss:Number(r.training_loss),validation:Number(r.validation_rmse??r.validation_reconstruction_mse)}))}><CartesianGrid stroke="#243746"/><XAxis dataKey="epoch"/><YAxis yAxisId="loss"/><YAxis yAxisId="validation" orientation="right"/><Tooltip/><Line yAxisId="loss" dataKey="loss" stroke="#99a6ff" dot={false}/><Line yAxisId="validation" dataKey="validation" stroke="#58e2b4" dot={false}/></LineChart></ResponsiveContainer></div>}<h2>Resolved hyperparameters</h2><pre>{JSON.stringify(selectedRun.config,null,2)}</pre><h2>Metrics and variability</h2><pre>{JSON.stringify(selectedRun.metrics,null,2)}</pre>{selectedRun.artifacts?.filter(a=>!a.startsWith('weights')).map(a=><p key={a}><a href={`${API}/experiments/${selectedRun.run_id}/artifacts/${a}`} target="_blank" rel="noreferrer">{a}</a></p>)}</section>}</>}
    <p className="footnote">Cycle-based predictions · Test truth excluded from model inputs · Models and thresholds are versioned · Simulated turbofan benchmark</p>
  </main></div>;
}
