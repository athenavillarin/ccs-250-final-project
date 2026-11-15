/** One method's output: before/after mask slider, opacity control, and stats, or a placeholder while it runs. */
import { useState } from "react";
import type { MethodResult } from "../api";

interface Props {
  result: MethodResult;
  imageUrl: string;
  aspect: number;
  recommended: boolean;
  pending?: boolean;
}

export default function ResultCard({ result, imageUrl, aspect, recommended, pending }: Props) {
  const [split, setSplit] = useState(100);
  const [opacity, setOpacity] = useState(0.6);

  return (
    <article
      className={`flex flex-col rounded-xl bg-white p-3 shadow-sm ring-1 ${
        recommended ? "ring-2 ring-sky-600" : "ring-slate-200"
      }`}
    >
      <header className="mb-2 flex items-center justify-between gap-2">
        <h3 className="font-semibold text-slate-900">{result.name ?? result.method}</h3>
        {recommended && <span className="rounded-full bg-sky-100 px-2 py-0.5 text-xs font-medium text-sky-800">Recommended</span>}
      </header>

      {pending ? (
        <div className="relative w-full overflow-hidden rounded-lg bg-slate-100" style={{ aspectRatio: aspect }}>
          <img src={imageUrl} alt="Input" className="absolute inset-0 h-full w-full object-fill opacity-40" />
          <p className="absolute inset-0 flex animate-pulse items-center justify-center text-sm font-medium text-slate-700">Running…</p>
        </div>
      ) : result.error ? (
        <p className="rounded-lg bg-red-50 p-3 text-sm text-red-700">Unavailable: {result.error}</p>
      ) : (
        <>
          <div className="relative w-full overflow-hidden rounded-lg bg-slate-100" style={{ aspectRatio: aspect }}>
            <img src={imageUrl} alt="Input" className="absolute inset-0 h-full w-full object-fill" />
            <img
              src={`data:image/png;base64,${result.mask_png}`}
              alt={`${result.name} flood mask`}
              className="absolute inset-0 h-full w-full object-fill"
              style={{ opacity, clipPath: `inset(0 ${100 - split}% 0 0)` }}
            />
            {split < 100 && <div className="absolute inset-y-0 w-0.5 bg-white shadow" style={{ left: `${split}%` }} />}
          </div>

          <label className="mt-3 text-xs text-slate-600">
            Before / after
            <input type="range" min={0} max={100} value={split} onChange={(e) => setSplit(+e.target.value)} className="w-full accent-sky-700" />
          </label>
          <label className="text-xs text-slate-600">
            Mask opacity
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={opacity}
              onChange={(e) => setOpacity(+e.target.value)}
              className="w-full accent-sky-700"
            />
          </label>

          <dl className="mt-2 grid grid-cols-3 gap-2 text-center">
            <Stat label="Flooded" value={`${result.flooded_pct?.toFixed(1)}%`} />
            <Stat label="Runtime" value={`${Math.round(result.runtime_ms ?? 0)} ms`} />
            <Stat label="Params" value={result.params_m ? `${result.params_m.toFixed(2)} M` : "0"} />
          </dl>
        </>
      )}
    </article>
  );
}

/** Small labeled number tile. */
function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-slate-50 py-1.5">
      <dt className="text-[11px] uppercase tracking-wide text-slate-500">{label}</dt>
      <dd className="text-sm font-semibold tabular-nums text-slate-900">{value}</dd>
    </div>
  );
}
