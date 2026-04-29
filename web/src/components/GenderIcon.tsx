import type { Gender } from "../types";

interface Props {
  gender: Gender;
  size?: number;
  className?: string;
}

// Chunky rounded icons. Filled circle with a rounded symbol on top.
// Keeps strokes thick (stroke-width 2.2 on a 24px canvas) so they read clearly at 14-20px.
export default function GenderIcon({ gender, size = 18, className }: Props) {
  if (!gender) return null;

  const common = {
    width: size,
    height: size,
    viewBox: "0 0 24 24",
    className,
    style: { display: "inline-block", verticalAlign: "middle" as const },
  };

  if (gender === "female") {
    return (
      <svg {...common} aria-label="female">
        <circle cx="12" cy="10" r="5.2" fill="#f4a0c0" stroke="#d94080" strokeWidth="2.2" />
        <line x1="12" y1="15.5" x2="12" y2="21" stroke="#d94080" strokeWidth="2.6" strokeLinecap="round" />
        <line x1="9.2" y1="18.6" x2="14.8" y2="18.6" stroke="#d94080" strokeWidth="2.6" strokeLinecap="round" />
      </svg>
    );
  }

  if (gender === "male") {
    return (
      <svg {...common} aria-label="male">
        <circle cx="10.5" cy="13.5" r="5.2" fill="#9dc2ff" stroke="#3b82f6" strokeWidth="2.2" />
        <line x1="14.2" y1="9.8" x2="20" y2="4" stroke="#3b82f6" strokeWidth="2.6" strokeLinecap="round" />
        <polyline points="15,4 20,4 20,9" fill="none" stroke="#3b82f6" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    );
  }

  // neutral — chunky gender-neutral circle with a dot
  return (
    <svg {...common} aria-label="neutral">
      <circle cx="12" cy="12" r="7.5" fill="#d6d6d6" stroke="#8a8a8a" strokeWidth="2.2" />
      <circle cx="12" cy="12" r="2.2" fill="#8a8a8a" />
    </svg>
  );
}
