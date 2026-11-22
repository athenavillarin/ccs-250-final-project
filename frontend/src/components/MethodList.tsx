/** Selectable list of the three methods with each one's status and detected water share. */
import { METHOD_INFO, type MethodKey, type MethodResult } from "../api";

interface Props {
  order: MethodKey[];
  results: Partial<Record<MethodKey, MethodResult>>;
  selected: MethodKey;
  recommended: MethodKey | null;
  onSelect: (key: MethodKey) => void;
}

export default function MethodList({ order, results, selected, recommended, onSelect }: Props) {
  return (
    <ul className="flex flex-col gap-2" role="radiogroup" aria-label="Method">
      {order.map((key) => {
        const r = results[key];
        const active = key === selected;
        return (
          <li key={key}>
            <button
              type="button"
              role="radio"
              aria-checked={active}
              onClick={() => onSelect(key)}
              className={`flex w-full items-center gap-3 rounded-xl px-4 py-3 text-left ring-1 transition-colors ${
                active ? "bg-sky-50 ring-2 ring-sky-600" : "bg-white ring-slate-200 hover:bg-slate-50"
              }`}
            >
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-semibold text-slate-900">{METHOD_INFO[key].name}</span>
                  {key === recommended && (
                    <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-xs font-medium text-emerald-800">
                      Most accurate
                    </span>
                  )}
                </div>
                <div className="text-xs text-slate-500">{METHOD_INFO[key].kind}</div>
              </div>
              <Status result={r} />
            </button>
          </li>
        );
      })}
    </ul>
  );
}

/** Water share and processing time once a method finishes, or its pending/error state. */
function Status({ result }: { result?: MethodResult }) {
  if (!result) return <span className="animate-pulse text-sm text-slate-500">Running…</span>;
  if (result.error) return <span className="text-sm font-medium text-red-700">Failed</span>;
  return (
    <span className="text-right">
      <span className="block text-lg font-semibold tabular-nums text-slate-900">{result.flooded_pct?.toFixed(0)}%</span>
      <span className="block text-xs tabular-nums text-slate-500">water · {((result.runtime_ms ?? 0) / 1000).toFixed(1)} s</span>
    </span>
  );
}
