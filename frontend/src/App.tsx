/** Flood Mapper: upload a photo, compare the three methods, and see the recommended one. */
import { useEffect, useState } from "react";
import { getHealth, getRecommendation, segment, type Health, type Recommendation, type SegmentResponse } from "./api";
import ResultCard from "./components/ResultCard";
import ResultsTable from "./components/ResultsTable";
import Uploader from "./components/Uploader";

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [rec, setRec] = useState<Recommendation | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [response, setResponse] = useState<SegmentResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch((e: Error) => setError(`Backend unreachable: ${e.message}`));
    getRecommendation()
      .then(setRec)
      .catch(() => setRec(null));
  }, []);

  const run = async (file: File) => {
    if (imageUrl) URL.revokeObjectURL(imageUrl);
    setImageUrl(URL.createObjectURL(file));
    setResponse(null);
    setError(null);
    setLoading(true);
    try {
      setResponse(await segment(file));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const recName = rec?.table.find((r) => r.method === rec.method)?.name;

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <main className="mx-auto max-w-6xl px-4 py-6 sm:py-10">
        <header className="mb-6">
          <h1 className="text-2xl font-bold sm:text-3xl">Flood Mapper</h1>
          <p className="mt-1 max-w-2xl text-slate-600">
            Map floodwater in drone and phone photos with three methods: a classical graph cut, a U-Net trained from scratch,
            and a pretrained SegFormer.
          </p>
          <p className="mt-2 text-xs text-slate-500">
            {health
              ? `Backend ${health.status} · ${health.device.toUpperCase()} · ${health.loaded.length}/3 models loaded`
              : "Connecting to backend…"}
          </p>
        </header>

        {rec?.method && (
          <section className="mb-6 rounded-xl bg-sky-50 p-4 ring-1 ring-sky-200">
            <h2 className="text-sm font-semibold text-sky-900">Recommended: {recName}</h2>
            <p className="mt-1 text-sm text-sky-900">{rec.explanation}</p>
          </section>
        )}

        <Uploader onFile={run} disabled={loading} />

        {error && <p className="mt-4 rounded-lg bg-red-50 p-3 text-sm text-red-700">{error}</p>}
        {loading && <p className="mt-4 animate-pulse text-sm text-slate-600">Running all three methods…</p>}

        {response && imageUrl && (
          <section className="mt-6 grid gap-4 md:grid-cols-3">
            {response.results.map((r) => (
              <ResultCard
                key={r.method}
                result={r}
                imageUrl={imageUrl}
                aspect={response.width / response.height}
                recommended={r.method === rec?.method}
              />
            ))}
          </section>
        )}

        <section className="mt-10 rounded-xl bg-white p-4 shadow-sm ring-1 ring-slate-200">
          <h2 className="font-semibold">How it works and test results</h2>
          <p className="mt-1 mb-3 text-sm text-slate-600">
            All methods use the same FloodNet train/val/test splits. Flood = water, flooded roads, and flooded buildings.
          </p>
          <ResultsTable rows={rec?.table ?? []} />
        </section>
      </main>
    </div>
  );
}
