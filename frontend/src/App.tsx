/** Flood Mapper: upload a photo or pick a labelled sample and compare the water each method finds. */
import { useEffect, useMemo, useRef, useState } from "react";
import { getHealth, getRecommended, segment, METHOD_INFO, type MethodKey, type MethodResult } from "./api";
import MethodList from "./components/MethodList";
import Samples, { type Sample } from "./components/Samples";
import Uploader from "./components/Uploader";
import Viewer, { type LegendItem } from "./components/Viewer";
import { AGREEMENT_COLORS, ERROR_COLORS, agreementOverlay, css, decodeMask, errorOverlay, miou, type Mask } from "./masks";

const ORDER: MethodKey[] = ["pretrained", "scratch", "classical"];
const WATER = "rgb(15 82 186)";

type Server = "checking" | "waking" | "ready" | "down";
type Mode = "mask" | "agreement" | "errors";

export default function App() {
  const [server, setServer] = useState<Server>("checking");
  const [recommended, setRecommended] = useState<MethodKey | null>(null);
  const [selected, setSelected] = useState<MethodKey>("pretrained");
  const [mode, setMode] = useState<Mode>("mask");
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [aspect, setAspect] = useState(4 / 3);
  const [sampleId, setSampleId] = useState<string | null>(null);
  const [truth, setTruth] = useState<Mask | null>(null);
  const [results, setResults] = useState<Partial<Record<MethodKey, MethodResult>>>({});
  const [masks, setMasks] = useState<Partial<Record<MethodKey, Mask>>>({});
  const current = useRef<AbortController | null>(null);

  useEffect(() => {
    const slow = setTimeout(() => setServer((s) => (s === "checking" ? "waking" : s)), 2500);
    getHealth()
      .then(() => setServer("ready"))
      .catch(() => setServer("down"))
      .finally(() => clearTimeout(slow));
    getRecommended()
      .then((key) => {
        setRecommended(key);
        if (key) setSelected(key);
      })
      .catch(() => setRecommended(null));
    return () => clearTimeout(slow);
  }, []);

  const run = async (file: File, sample?: Sample) => {
    current.current?.abort();
    const controller = new AbortController();
    current.current = controller;
    if (imageUrl) URL.revokeObjectURL(imageUrl);
    const url = URL.createObjectURL(file);
    const ratio = await imageAspect(url);
    const labels = sample ? await decodeMask(sample.labels) : null;
    if (controller.signal.aborted) return;
    setAspect(ratio);
    setImageUrl(url);
    setSampleId(sample?.id ?? null);
    setTruth(labels);
    setResults({});
    setMasks({});
    setMode((m) => (m === "errors" && !labels ? "mask" : m));
    for (const key of ORDER) {
      try {
        const result = await segment(file, key, controller.signal);
        const mask = result.mask_png
          ? await decodeMask(`data:image/png;base64,${result.mask_png}`, labels?.width, labels?.height)
          : undefined;
        if (controller.signal.aborted) return;
        setResults((prev) => ({ ...prev, [key]: result }));
        if (mask) setMasks((prev) => ({ ...prev, [key]: mask }));
      } catch (e) {
        if (controller.signal.aborted) return;
        setResults((prev) => ({ ...prev, [key]: { method: key, error: (e as Error).message } }));
      }
    }
  };

  /** Return to the start screen: stop any running request and clear the photo and its results. */
  const startOver = () => {
    current.current?.abort();
    current.current = null;
    if (imageUrl) URL.revokeObjectURL(imageUrl);
    setImageUrl(null);
    setSampleId(null);
    setTruth(null);
    setResults({});
    setMasks({});
    setMode((m) => (m === "errors" ? "mask" : m));
  };

  const pickSample = async (sample: Sample) => {
    const blob = await (await fetch(sample.photo)).blob();
    run(new File([blob], `${sample.id}.jpg`, { type: blob.type }), sample);
  };

  const scores = useMemo(() => {
    const out: Partial<Record<MethodKey, number>> = {};
    if (truth) for (const key of ORDER) if (masks[key]) out[key] = miou(masks[key]!, truth);
    return out;
  }, [masks, truth]);

  const view = useMemo(() => {
    const name = METHOD_INFO[selected].name;
    const result = results[selected];
    const failed = result?.error ? `${name} could not process this photo: ${result.error}` : undefined;
    if (mode === "agreement") {
      const done = ORDER.every((k) => results[k]);
      const available = ORDER.flatMap((k) => (masks[k] ? [masks[k]!] : []));
      const legend: LegendItem[] = available.map((_, n) => ({
        color: css(AGREEMENT_COLORS[n]),
        label: n + 1 === available.length ? `All ${n + 1} methods` : `${n + 1} method${n ? "s" : ""}`,
      }));
      return {
        overlay: done && available.length ? agreementOverlay(available) : undefined,
        legend,
        waiting: "Waiting for all three methods…",
        note: "Darker blue means more methods marked that spot as water.",
      };
    }
    if (mode === "errors" && truth) {
      return {
        overlay: masks[selected] ? errorOverlay(masks[selected]!, truth) : undefined,
        legend: [
          { color: css(ERROR_COLORS.hit), label: "Correct water" },
          { color: css(ERROR_COLORS.missed), label: "Missed water" },
          { color: css(ERROR_COLORS.false), label: "False alarm" },
        ],
        waiting: `Running ${name}…`,
        error: failed,
        note: `${name} compared with the water outlined by hand.`,
      };
    }
    return {
      overlay: result?.mask_png ? `data:image/png;base64,${result.mask_png}` : undefined,
      legend: [{ color: WATER, label: "Water" }],
      waiting: `Running ${name}…`,
      error: failed,
      download: result?.mask_png
        ? { url: `data:image/png;base64,${result.mask_png}`, name: `water-mask-${selected}.png` }
        : undefined,
    };
  }, [mode, selected, results, masks, truth]);

  const modes: [Mode, string][] = [
    ["mask", "Water mask"],
    ["agreement", "Agreement"],
    ...(truth ? ([["errors", "Errors vs. labels"]] as [Mode, string][]) : []),
  ];

  return (
    <div className="flex min-h-screen flex-col bg-gray-50 text-gray-900">
      <header className="border-b border-gray-200 bg-white">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-3">
          <span className="flex items-center gap-2 font-semibold text-gray-900">
            <svg viewBox="0 0 24 24" className="h-5 w-5 text-brand" fill="currentColor" aria-hidden>
              <path d="M12 2.5c-.3 0-.6.2-.8.4C9.6 5 6 9.7 6 13.5a6 6 0 0 0 12 0c0-3.8-3.6-8.5-5.2-10.6-.2-.2-.5-.4-.8-.4Z" />
            </svg>
            Flood Mapper
          </span>
          {imageUrl && (
            <div className="flex items-center gap-3">
              <button
                type="button"
                onClick={startOver}
                className="text-sm font-semibold text-gray-600 hover:text-gray-900 hover:underline"
              >
                Start over
              </button>
              <Uploader onFile={run} compact />
            </div>
          )}
        </div>
      </header>

      <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-8 sm:py-12">
        {server === "waking" && (
          <p className="mb-4 rounded border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
            Starting the server. This can take up to a minute after a quiet period.
          </p>
        )}
        {server === "down" && (
          <p className="mb-4 rounded border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            The server is not responding. Reload the page in a minute.
          </p>
        )}

        {!imageUrl ? (
          <section className="mx-auto max-w-2xl">
            <h1 className="text-xl font-semibold text-gray-900">Find the floodwater in a photo</h1>
            <p className="mt-2 mb-8 text-sm text-gray-500">
              Upload a street, drone, or aerial photo of a flood. Three methods mark the water so you can compare them.
            </p>
            <Uploader onFile={run} disabled={server === "down"} />
            <h2 className="mt-8 mb-2 text-sm font-semibold text-gray-600">Or try a sample photo with hand-drawn labels</h2>
            <Samples onPick={pickSample} />
          </section>
        ) : (
          <section className="grid gap-8 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
            <div>
              <div className="mb-4 flex flex-wrap gap-1 rounded bg-gray-100 p-1" role="tablist" aria-label="View">
                {modes.map(([key, label]) => (
                  <button
                    key={key}
                    type="button"
                    role="tab"
                    aria-selected={mode === key}
                    onClick={() => setMode(key)}
                    className={`flex-1 rounded px-3 py-1.5 text-sm font-semibold whitespace-nowrap transition-colors ${
                      mode === key ? "bg-white text-gray-900 shadow-sm" : "text-gray-500 hover:text-gray-900"
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>
              <Viewer imageUrl={imageUrl} aspect={aspect} {...view} />
              {view.note && view.overlay && <p className="mt-2 text-sm text-gray-500">{view.note}</p>}
            </div>
            <div>
              <h2 className="mb-2 text-sm font-semibold text-gray-500">Methods</h2>
              <MethodList
                order={ORDER}
                results={results}
                selected={selected}
                recommended={recommended}
                scores={scores}
                onSelect={(key) => {
                  setSelected(key);
                  setMode((m) => (m === "agreement" ? "mask" : m));
                }}
              />
              <p className="mt-4 text-sm text-gray-400">Water share is the percentage of the photo marked as water.</p>
              <h2 className="mt-8 mb-2 text-sm font-semibold text-gray-500">Sample photos</h2>
              <Samples onPick={pickSample} active={sampleId ?? undefined} />
            </div>
          </section>
        )}
      </main>

      <footer className="py-8 text-center text-sm text-gray-400">Photos are processed on the server and are not stored.</footer>
    </div>
  );
}

/** Resolve an image URL's width-to-height ratio. */
function imageAspect(url: string): Promise<number> {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve(img.naturalWidth / img.naturalHeight);
    img.onerror = () => resolve(4 / 3);
    img.src = url;
  });
}
