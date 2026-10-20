export const API = process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://localhost:8000/api/v1';
export async function api<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(API + path, body === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error?.message ?? `Request failed (${response.status})`);
  return data as T;
}
export function formatMetric(value: number | null | undefined): string {
  return value == null || !Number.isFinite(value) ? 'Unavailable' : value.toFixed(1);
}
export type Prediction = {cycle:number; status:'ready'|'warming_up'; rul_cycles:number|null; raw_rul_cycles:number|null; anomaly_score:number|null; anomaly_threshold:number|null; health_index:number|null; recommendation:string|null; reason_codes:string[]; model_version:string; source:string};
export type Machine = {machine_id:string; engine_id:number; partition:string; available_cycles:number};
export type Run = {run_id:string; model_version:string; family:string; status?:string; config:{seed:number; window_length:number; rul_cap_cycles:number|null; [key:string]:unknown}; metrics?:{validation?:{mae_cycles:number;rmse_cycles:number;nasa_score_sum:number};healthy_proxy_validation_mse?:number}; ['history.csv']?:Record<string,string>[]; ['predictions.csv']?:Record<string,string>[]; artifacts?:string[]};
export type Session = {session_id:string;generation:number;cycle:number;status:'paused'|'running'|'completed';model_version:string;last_prediction:Prediction|null;prediction?:Prediction; speed_cps:number};
