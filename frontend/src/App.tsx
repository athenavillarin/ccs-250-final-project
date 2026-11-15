/** App shell: shows backend status until the upload and result views land. */
import { useEffect, useState } from "react";
import { getHealth, type Health } from "./api";

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getHealth().then(setHealth).catch((e: Error) => setError(e.message));
  }, []);

  return (
    <main className="mx-auto max-w-5xl p-4">
      <h1 className="text-2xl font-semibold">Flood Mapper</h1>
      <p className="text-sm text-gray-600">
        {error ? `Backend unreachable: ${error}` : health ? `Backend ${health.status} on ${health.device}: ${health.loaded.join(", ")}` : "Connecting…"}
      </p>
    </main>
  );
}
