/** Selectable list of the three methods with each one's status, detected water share, and score on labelled samples. */
import { METHOD_INFO, type MethodKey, type MethodResult } from "../api";

interface Props {
  order: MethodKey[];
  results: Partial<Record<MethodKey, MethodResult>>;
  selected: MethodKey;
  recommended: MethodKey | null;
  scores: Partial<Record<MethodKey, number>>;
  onSelect: (key: MethodKey) => void;
}

export default function MethodList({ order, results, selected, recommended, scores, onSelect }: Props) {
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
              className={`flex w-full items-center gap-3 rounded px-4 py-3 text-left border transition-colors ${
                active ? "bg-brand-light border-brand" : "bg-white border-gray-200 hover:bg-gray-50"
              }`}
            >
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-semibold text-gray-900">{METHOD_INFO[key].name}</span>
                  {key === recommended && (
                    <span className="rounded bg-brand-light px-2 py-0.5 text-sm text-brand-dark font-semibold">
                      Most accurate
                    </span>
                  )}
                </div>
                <div className="text-sm text-gray-500">{METHOD_INFO[key].kind}</div>
                {scores[key] !== undefined && (
                  <div className="mt-1 text-sm font-semibold text-gray-700">
                    Score vs. labels: <span className="tabular-nums">{scores[key]!.toFixed(0)}%</span> mIoU
                  </div>
                )}
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
  if (!result) return <span className="animate-pulse text-sm text-gray-400">Running...</span>;
  if (result.error) return <span className="text-sm font-semibold text-red-600">Failed</span>;
  return (
    <span className="text-right">
      <span className="block text-base font-semibold tabular-nums text-gray-900">{result.flooded_pct?.toFixed(0)}%</span>
      <span className="block text-sm tabular-nums text-gray-400">water · {((result.runtime_ms ?? 0) / 1000).toFixed(1)} s</span>
    </span>
  );
}
