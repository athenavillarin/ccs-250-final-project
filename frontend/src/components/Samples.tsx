/** Row of sample flood photos that come with hand-drawn water labels. */
export interface Sample {
  id: string;
  photo: string;
  labels: string;
}

const IDS = ["ph_003", "ph_004", "ph_006", "ph_008", "ph_010"];

export const SAMPLES: Sample[] = IDS.map((id) => ({
  id,
  photo: `/samples/${id}.jpg`,
  labels: `/samples/${id}_mask.png`,
}));

interface Props {
  onPick: (sample: Sample) => void;
  active?: string;
}

export default function Samples({ onPick, active }: Props) {
  return (
    <div>
      <div className="flex gap-2 overflow-x-auto pb-1">
        {SAMPLES.map((s) => (
          <button
            key={s.id}
            type="button"
            onClick={() => onPick(s)}
            aria-label={`Sample photo ${s.id}`}
            className={`shrink-0 overflow-hidden rounded border transition hover:opacity-90 ${
              s.id === active ? "border-2 border-brand" : "border-gray-200"
            }`}
          >
            <img src={`/samples/${s.id}_thumb.jpg`} alt="" loading="lazy" className="h-16 w-[5.33rem] object-cover sm:h-20 sm:w-[6.67rem]" />
          </button>
        ))}
      </div>
      <p className="mt-2 text-sm text-gray-400">
        Flooding in Calumpit, Bulacan. Photos by Judgefloro (CC0, Wikimedia Commons); water outlines drawn by our team.
      </p>
    </div>
  );
}
