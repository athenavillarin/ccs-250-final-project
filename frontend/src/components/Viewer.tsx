/** Large photo view with a colour overlay, a draggable before/after divider, a legend, and overlay controls. */
import { useRef, useState, type PointerEvent } from "react";

export interface LegendItem {
  color: string;
  label: string;
}

interface Props {
  imageUrl: string;
  aspect: number;
  overlay?: string;
  legend: LegendItem[];
  waiting?: string;
  error?: string;
  download?: { url: string; name: string };
}

export default function Viewer({ imageUrl, aspect, overlay, legend, waiting, error, download }: Props) {
  const [split, setSplit] = useState(100);
  const [opacity, setOpacity] = useState(0.6);
  const box = useRef<HTMLDivElement>(null);

  const drag = (e: PointerEvent) => {
    if (!box.current || (e.type === "pointermove" && e.buttons !== 1)) return;
    const rect = box.current.getBoundingClientRect();
    setSplit(Math.min(100, Math.max(0, ((e.clientX - rect.left) / rect.width) * 100)));
  };

  return (
    <div className="mx-auto" style={{ width: `min(100%, calc(72vh * ${aspect}))` }}>
      <div
        ref={box}
        onPointerDown={overlay ? drag : undefined}
        onPointerMove={overlay ? drag : undefined}
        className={`relative w-full touch-none select-none overflow-hidden rounded-xl bg-slate-200 ${overlay ? "cursor-ew-resize" : ""}`}
        style={{ aspectRatio: aspect }}
      >
        <img src={imageUrl} alt="Uploaded photo" draggable={false} className="absolute inset-0 h-full w-full object-fill" />
        {overlay && (
          <img
            src={overlay}
            alt="Overlay"
            draggable={false}
            className="absolute inset-0 h-full w-full object-fill"
            style={{ opacity, clipPath: `inset(0 ${100 - split}% 0 0)` }}
          />
        )}
        {overlay && split < 100 && (
          <div className="pointer-events-none absolute inset-y-0" style={{ left: `${split}%` }}>
            <div className="h-full w-0.5 -translate-x-1/2 bg-white shadow" />
            <div className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-white px-2 py-1 text-xs font-medium text-slate-700 shadow">
              ⟷
            </div>
          </div>
        )}
        {waiting && !overlay && (
          <div className="absolute inset-0 flex items-center justify-center bg-white/50">
            <p className="animate-pulse rounded-full bg-white px-4 py-2 text-sm font-medium text-slate-700 shadow">{waiting}</p>
          </div>
        )}
        {error && (
          <div className="absolute inset-0 flex items-center justify-center bg-white/70 p-6">
            <p className="rounded-lg bg-red-50 px-4 py-3 text-sm text-red-700 ring-1 ring-red-200">{error}</p>
          </div>
        )}
      </div>

      {overlay && (
        <div className="mt-3 flex flex-wrap items-center gap-x-6 gap-y-3 text-sm text-slate-600">
          <span className="flex flex-wrap items-center gap-x-4 gap-y-1">
            {legend.map((item) => (
              <span key={item.label} className="flex items-center gap-1.5">
                <span className="h-3 w-3 rounded-sm" style={{ background: item.color }} aria-hidden />
                {item.label}
              </span>
            ))}
          </span>
          <label className="flex items-center gap-2">
            Overlay
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={opacity}
              onChange={(e) => setOpacity(+e.target.value)}
              className="w-24 accent-sky-700"
            />
          </label>
          <button type="button" onClick={() => setSplit(split < 100 ? 100 : 50)} className="font-medium text-sky-800 hover:underline">
            {split < 100 ? "Hide comparison" : "Compare with original"}
          </button>
          {download && (
            <a href={download.url} download={download.name} className="font-medium text-sky-800 hover:underline sm:ml-auto">
              Download mask
            </a>
          )}
        </div>
      )}
    </div>
  );
}
