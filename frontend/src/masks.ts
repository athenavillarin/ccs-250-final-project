/** Browser-side mask helpers: decode PNG masks to pixels, score against labels, and paint colour overlays. */

export interface Mask {
  width: number;
  height: number;
  water: Uint8Array;
}

type Rgb = [number, number, number];

export const ERROR_COLORS: Record<"hit" | "missed" | "false", Rgb> = {
  hit: [15, 82, 186],
  missed: [245, 158, 11],
  false: [217, 70, 239],
};

export const AGREEMENT_COLORS: Rgb[] = [
  [147, 197, 253],
  [59, 130, 246],
  [30, 58, 138],
];

/** CSS colour string for an RGB triple. */
export const css = ([r, g, b]: Rgb) => `rgb(${r} ${g} ${b})`;

/** Load an image element from a URL. */
function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`could not load ${src}`));
    img.src = src;
  });
}

/** Decode a mask PNG (coloured-on-transparent or white-on-black) to 0/1 pixels, optionally resampled to a size. */
export async function decodeMask(src: string, width?: number, height?: number): Promise<Mask> {
  const img = await loadImage(src);
  const w = width ?? img.naturalWidth;
  const h = height ?? img.naturalHeight;
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d")!;
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(img, 0, 0, w, h);
  const rgba = ctx.getImageData(0, 0, w, h).data;
  const water = new Uint8Array(w * h);
  for (let i = 0; i < water.length; i++) {
    const p = 4 * i;
    water[i] = rgba[p + 3] > 127 && Math.max(rgba[p], rgba[p + 1], rgba[p + 2]) > 127 ? 1 : 0;
  }
  return { width: w, height: h, water };
}

/** Mean of water IoU and non-water IoU, in percent, matching the report's mIoU. */
export function miou(pred: Mask, truth: Mask): number {
  let tp = 0, fp = 0, fn = 0, tn = 0;
  for (let i = 0; i < truth.water.length; i++) {
    const p = pred.water[i], t = truth.water[i];
    if (p && t) tp++;
    else if (p) fp++;
    else if (t) fn++;
    else tn++;
  }
  const water = tp + fp + fn ? tp / (tp + fp + fn) : 1;
  const dry = tn + fp + fn ? tn / (tn + fp + fn) : 1;
  return 50 * (water + dry);
}

/** Render a transparent PNG where colourOf returns a colour for a pixel index, or null to leave it clear. */
function paint(width: number, height: number, colourOf: (i: number) => Rgb | null): string {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d")!;
  const out = ctx.createImageData(width, height);
  for (let i = 0; i < width * height; i++) {
    const c = colourOf(i);
    if (!c) continue;
    out.data.set([c[0], c[1], c[2], 255], 4 * i);
  }
  ctx.putImageData(out, 0, 0);
  return canvas.toDataURL("image/png");
}

/** Overlay colouring correct water, missed water, and false alarms of a prediction against labels. */
export const errorOverlay = (pred: Mask, truth: Mask) =>
  paint(truth.width, truth.height, (i) =>
    pred.water[i] && truth.water[i] ? ERROR_COLORS.hit : truth.water[i] ? ERROR_COLORS.missed : pred.water[i] ? ERROR_COLORS.false : null,
  );

/** Overlay shading each pixel by how many of the given masks call it water. */
export function agreementOverlay(masks: Mask[]): string {
  const { width, height } = masks[0];
  return paint(width, height, (i) => {
    let n = 0;
    for (const m of masks) n += m.water[i];
    return n ? AGREEMENT_COLORS[n - 1] : null;
  });
}
