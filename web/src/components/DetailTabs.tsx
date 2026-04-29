import { useState, type ReactNode } from "react";

export type DetailTabKey = "profile" | "f0" | "mel";

interface Props {
  children: Record<DetailTabKey, ReactNode>;
  initial?: DetailTabKey;
}

const LABELS: Record<DetailTabKey, string> = {
  profile: "Profile",
  f0: "F0 contour",
  mel: "Mel-spectrogram",
};

export default function DetailTabs({ children, initial = "profile" }: Props) {
  const [active, setActive] = useState<DetailTabKey>(initial);
  const [opened, setOpened] = useState<Set<DetailTabKey>>(new Set([initial]));
  const keys: DetailTabKey[] = ["profile", "f0", "mel"];

  const activate = (k: DetailTabKey) => {
    setActive(k);
    if (!opened.has(k)) setOpened(new Set(opened).add(k));
  };

  return (
    <div className="detail-tabs">
      <div className="tab-nav">
        {keys.map((k) => (
          <button
            key={k}
            className={active === k ? "active" : ""}
            onClick={() => activate(k)}
          >{LABELS[k]}</button>
        ))}
      </div>
      <div className="detail-tab-body">
        {keys.map((k) =>
          opened.has(k) ? (
            <div key={k} style={{ display: active === k ? "block" : "none" }}>
              {children[k]}
            </div>
          ) : null,
        )}
      </div>
    </div>
  );
}
