/** Test-set comparison table read from the backend's results/ files. */
import type { ResultRow } from "../api";

const fmt = (v: string, digits = 1) => (v === "" || isNaN(+v) ? "—" : (+v).toFixed(digits));

export default function ResultsTable({ rows }: { rows: ResultRow[] }) {
  if (!rows.length) return <p className="text-sm text-slate-500">Results will appear once the experiments have run.</p>;
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[560px] text-sm">
        <thead>
          <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500">
            <th className="py-2 pr-3">Method</th>
            <th className="py-2 pr-3 text-right">mIoU (%)</th>
            <th className="py-2 pr-3 text-right">Dice</th>
            <th className="py-2 pr-3 text-right">Pixel acc (%)</th>
            <th className="py-2 pr-3 text-right">Runtime (ms)</th>
            <th className="py-2 pr-3 text-right">Params (M)</th>
            <th className="py-2 text-right">Train (min)</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.method} className="border-b border-slate-100 tabular-nums">
              <td className="py-2 pr-3">
                <div className="font-medium text-slate-900">{r.name}</div>
                <div className="text-xs text-slate-500">{r.slot}</div>
              </td>
              <td className="py-2 pr-3 text-right">{fmt(r.miou)}</td>
              <td className="py-2 pr-3 text-right">{fmt(r.dice, 3)}</td>
              <td className="py-2 pr-3 text-right">{fmt(r.pixel_acc)}</td>
              <td className="py-2 pr-3 text-right">
                {fmt(r.runtime_ms, 0)} <span className="text-xs text-slate-500">{r.device}</span>
              </td>
              <td className="py-2 pr-3 text-right">{fmt(r.params_m, 2)}</td>
              <td className="py-2 text-right">{fmt(r.train_min)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
