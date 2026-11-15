/** Drag-and-drop image picker with a camera shortcut on mobile. */
import { useRef, useState, type DragEvent } from "react";

interface Props {
  onFile: (file: File) => void;
  disabled?: boolean;
}

export default function Uploader({ onFile, disabled }: Props) {
  const fileInput = useRef<HTMLInputElement>(null);
  const cameraInput = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const pick = (files: FileList | null) => {
    const file = files?.[0];
    if (file && file.type.startsWith("image/")) onFile(file);
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    if (!disabled) pick(e.dataTransfer.files);
  };

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      className={`rounded-xl border-2 border-dashed p-6 text-center transition-colors ${
        dragging ? "border-sky-500 bg-sky-50" : "border-slate-300 bg-white"
      }`}
    >
      <p className="text-slate-700">Drop a drone or phone photo here</p>
      <p className="mt-1 text-xs text-slate-500">JPG or PNG, up to 10 MB</p>
      <div className="mt-4 flex flex-wrap justify-center gap-2">
        <button
          type="button"
          disabled={disabled}
          onClick={() => fileInput.current?.click()}
          className="rounded-lg bg-sky-700 px-4 py-2 text-sm font-medium text-white hover:bg-sky-800 disabled:opacity-50"
        >
          Choose image
        </button>
        <button
          type="button"
          disabled={disabled}
          onClick={() => cameraInput.current?.click()}
          className="rounded-lg border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50 sm:hidden"
        >
          Take photo
        </button>
      </div>
      <input ref={fileInput} type="file" accept="image/*" hidden onChange={(e) => pick(e.target.files)} />
      <input ref={cameraInput} type="file" accept="image/*" capture="environment" hidden onChange={(e) => pick(e.target.files)} />
    </div>
  );
}
