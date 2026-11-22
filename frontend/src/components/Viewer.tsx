/** Large photo view with the selected method's water mask, a draggable before/after divider, and mask controls. */
import { useRef, useState, type PointerEvent } from "react";
import type { MethodResult } from "../api";

interface Props {
  result?: MethodResult;
  name: string;
  imageUrl: string;
  aspect: number;
}

export default function Viewer({ result, name, imageUrl, aspect }: Props) {
  const [split, setSplit] = useState(100);
  const [opacity, setOpacity] = useState(0.6);
  const box = useRef<HTMLDivElement>(null);
  const ready = result?.mask_png;

  const drag = (e: PointerEvent) => {
    if (!box.current || (e.type === "pointermove" && e.buttons !== 1)) return;
    const rect = box.current.getBoundingClientRect();
    setSplit(Math.min(100, Math.max(0, ((e.clientX - rect.left) / rect.width) * 100)));
  };

  return (
    <div className="mx-auto" style={{ width: `min(100%, calc(72vh * ${aspect}))` }}>
      <div
        ref={box}
        onPointerDown={ready ? drag : undefined}
        onPointerMove={ready ? drag : undefined}
        className={`relative w-full touch-none select-none overflow-hidden rounded-xl bg-slate-200 ${ready ? "cursor-ew-resize" : ""}`}
        style={{ aspectRatio: aspect }}
      >
        <img src={imageUrl} alt="Uploaded photo" draggable={false} className="absolute inset-0 h-full w-full object-fill" />
        {ready && (
          <img
            src={`data:image/png;base64,${result.mask_png}`}
            alt={`Water detected by ${name}`}
            draggable={false}
            className="absolute inset-0 h-full w-full object-fill"
            style={{ opacity, clipPath: `inset(0 ${100 - split}% 0 0)` }}
          />
        )}
        {ready && split < 100 && (
          <div className="pointer-events-none absolute inset-y-0" style={{ left: `${split}%` }}>
            <div className="h-full w-0.5 -translate-x-1/2 bg-white shadow" />
            <div className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-white px-2 py-1 text-xs font-medium text-slate-700 shadow">
              ⟷
            </div>
          </div>
        )}
        {!result && (
          <div className="absolute inset-0 flex items-center justify-center bg-white/50">
            <p className="animate-pulse rounded-full bg-white px-4 py-2 text-sm font-medium text-slate-700 shadow">
              Running {name}…
            </p>
          </div>
        )}
        {result?.error && (
          <div className="absolute inset-0 flex items-center justify-center bg-white/70 p-6">
            <p className="rounded-lg bg-red-50 px-4 py-3 text-sm text-red-700 ring-1 ring-red-200">
              {name} could not process this photo: {result.error}
            </p>
          </div>
        )}
      </div>

      {ready && (
        <div className="mt-3 flex flex-wrap items-center gap-x-6 gap-y-3 text-sm text-slate-600">
          <span className="flex items-center gap-2">
            <span className="h-3 w-3 rounded-sm bg-[#0078ff]" aria-hidden />
            Water
          </span>
          <label className="flex items-center gap-2">
            Mask
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={opacity}
              onChange={(e) => setOpacity(+e.target.value)}
              className="w-28 accent-sky-700"
            />
          </label>
          <button type="button" onClick={() => setSplit(split < 100 ? 100 : 50)} className="font-medium text-sky-800 hover:underline">
            {split < 100 ? "Hide comparison" : "Compare with original"}
          </button>
          <a
            href={`data:image/png;base64,${result.mask_png}`}
            download={`water-mask-${result.method}.png`}
            className="font-medium text-sky-800 hover:underline sm:ml-auto"
          >
            Download mask
          </a>
        </div>
      )}
    </div>
  );
}
