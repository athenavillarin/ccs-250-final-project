/** Flood Mapper: upload a photo and compare the water each method finds. */
import { useEffect, useState } from "react";
import { getHealth, getRecommended, segment, METHOD_INFO, type MethodKey, type MethodResult } from "./api";
import MethodList from "./components/MethodList";
import Uploader from "./components/Uploader";
import Viewer from "./components/Viewer";

const ORDER: MethodKey[] = ["pretrained", "scratch", "classical"];

type Server = "checking" | "waking" | "ready" | "down";

export default function App() {
  const [server, setServer] = useState<Server>("checking");
  const [recommended, setRecommended] = useState<MethodKey | null>(null);
  const [selected, setSelected] = useState<MethodKey>("pretrained");
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [aspect, setAspect] = useState(4 / 3);
  const [results, setResults] = useState<Partial<Record<MethodKey, MethodResult>>>({});
  const [running, setRunning] = useState(false);

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

  const run = async (file: File) => {
    if (imageUrl) URL.revokeObjectURL(imageUrl);
    const url = URL.createObjectURL(file);
    setAspect(await imageAspect(url));
    setImageUrl(url);
    setResults({});
    setRunning(true);
    for (const key of ORDER) {
      try {
        const result = await segment(file, key);
        setResults((prev) => ({ ...prev, [key]: result }));
      } catch (e) {
        setResults((prev) => ({ ...prev, [key]: { method: key, error: (e as Error).message } }));
      }
    }
    setRunning(false);
  };

  return (
    <div className="flex min-h-screen flex-col bg-slate-50 text-slate-900">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-3">
          <span className="flex items-center gap-2 font-semibold">
            <svg viewBox="0 0 24 24" className="h-6 w-6 text-sky-700" fill="currentColor" aria-hidden>
              <path d="M12 2.5c-.3 0-.6.2-.8.4C9.6 5 6 9.7 6 13.5a6 6 0 0 0 12 0c0-3.8-3.6-8.5-5.2-10.6-.2-.2-.5-.4-.8-.4Z" />
            </svg>
            Flood Mapper
          </span>
          {imageUrl && <Uploader onFile={run} disabled={running} compact />}
        </div>
      </header>

      <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-6 sm:py-10">
        {server === "waking" && (
          <p className="mb-4 rounded-lg bg-amber-50 px-4 py-3 text-sm text-amber-900 ring-1 ring-amber-200">
            Starting the server. This can take up to a minute after a quiet period.
          </p>
        )}
        {server === "down" && (
          <p className="mb-4 rounded-lg bg-red-50 px-4 py-3 text-sm text-red-700 ring-1 ring-red-200">
            The server is not responding. Reload the page in a minute.
          </p>
        )}

        {!imageUrl ? (
          <section className="mx-auto max-w-2xl">
            <h1 className="text-2xl font-bold sm:text-3xl">Find the floodwater in a photo</h1>
            <p className="mt-2 mb-6 text-slate-600">
              Upload a street, drone, or aerial photo of a flood. Three methods mark the water so you can compare them.
            </p>
            <Uploader onFile={run} disabled={server === "down"} />
          </section>
        ) : (
          <section className="grid gap-6 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
            <Viewer result={results[selected]} name={METHOD_INFO[selected].name} imageUrl={imageUrl} aspect={aspect} />
            <div>
              <h2 className="mb-2 text-sm font-semibold text-slate-500">Methods</h2>
              <MethodList order={ORDER} results={results} selected={selected} recommended={recommended} onSelect={setSelected} />
              <p className="mt-3 text-xs text-slate-500">Water share is the percentage of the photo marked as water.</p>
            </div>
          </section>
        )}
      </main>

      <footer className="py-6 text-center text-xs text-slate-500">Photos are processed on the server and are not stored.</footer>
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
