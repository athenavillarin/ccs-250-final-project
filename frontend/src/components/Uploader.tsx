/** Photo picker: a large drop zone for the empty state, or a compact "New photo" button. */
import { useRef, useState, type DragEvent } from "react";

interface Props {
  onFile: (file: File) => void;
  disabled?: boolean;
  compact?: boolean;
}

export default function Uploader({ onFile, disabled, compact }: Props) {
  const fileInput = useRef<HTMLInputElement>(null);
  const cameraInput = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const pick = (input: HTMLInputElement) => {
    const file = input.files?.[0];
    input.value = "";
    if (file && file.type.startsWith("image/")) onFile(file);
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (!disabled && file?.type.startsWith("image/")) onFile(file);
  };

  const inputs = (
    <>
      <input ref={fileInput} type="file" accept="image/*" hidden onChange={(e) => pick(e.currentTarget)} />
      <input ref={cameraInput} type="file" accept="image/*" capture="environment" hidden onChange={(e) => pick(e.currentTarget)} />
    </>
  );

  if (compact) {
    return (
      <>
        <button
          type="button"
          disabled={disabled}
          onClick={() => fileInput.current?.click()}
          className="rounded border border-gray-300 bg-white px-3 py-1.5 text-sm font-semibold text-gray-700 hover:bg-gray-50 disabled:opacity-50"
        >
          New photo
        </button>
        {inputs}
      </>
    );
  }

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      className={`flex flex-col items-center rounded border-2 border-dashed px-8 py-16 text-center transition-colors ${
        dragging ? "border-brand bg-brand-light" : "border-gray-300 bg-white"
      }`}
    >
      <svg viewBox="0 0 24 24" className="h-8 w-8 text-brand" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden>
        <path d="M3 16.5v1.5A2.25 2.25 0 0 0 5.25 20.25h13.5A2.25 2.25 0 0 0 21 18v-1.5M16.5 8.25 12 3.75m0 0L7.5 8.25M12 3.75v12" />
      </svg>
      <p className="mt-4 font-semibold text-gray-800">
        <span className="hidden sm:inline">Drag a photo here, or choose one</span>
        <span className="sm:hidden">Choose a photo or take one</span>
      </p>
      <p className="mt-1 text-sm text-gray-500">JPG or PNG, up to 10 MB</p>
      <div className="mt-8 flex flex-wrap justify-center gap-2">
        <button
          type="button"
          disabled={disabled}
          onClick={() => fileInput.current?.click()}
          className="rounded bg-brand px-4 py-2 text-sm font-semibold text-white hover:bg-brand-dark disabled:opacity-50"
        >
          Choose photo
        </button>
        <button
          type="button"
          disabled={disabled}
          onClick={() => cameraInput.current?.click()}
          className="rounded border border-gray-300 px-4 py-2 text-sm font-semibold text-gray-700 hover:bg-gray-50 disabled:opacity-50 sm:hidden"
        >
          Take photo
        </button>
      </div>
      {inputs}
    </div>
  );
}
