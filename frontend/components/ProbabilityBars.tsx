"use client";

// PRD 7.7 — the 4-class distribution. Every bar labeled with NOVA class
// name + %, predicted class highlighted: this is what makes model
// ambiguity visible (e.g. a 0.87 / 0.12 split instead of a silent single
// number). Neutral ramp for highlighting — never green/red (PRD binding
// colour rule).

import type { NovaClass } from "@/types/predict";
import { NOVA_LABELS } from "@/types/predict";

interface ProbabilityBarsProps {
  probabilities: Record<"1" | "2" | "3" | "4", number>;
  predicted: NovaClass;
}

const NOVA_VAR: Record<NovaClass, { bg: string; fg: string }> = {
  1: { bg: "var(--nova-1)", fg: "var(--nova-1-fg)" },
  2: { bg: "var(--nova-2)", fg: "var(--nova-2-fg)" },
  3: { bg: "var(--nova-3)", fg: "var(--nova-3-fg)" },
  4: { bg: "var(--nova-4)", fg: "var(--nova-4-fg)" },
};

export default function ProbabilityBars({
  probabilities,
  predicted,
}: ProbabilityBarsProps) {
  const classes: NovaClass[] = [1, 2, 3, 4];

  return (
    <ul className="flex flex-col gap-2.5" aria-label="Class probabilities">
      {classes.map((cls) => {
        const p =
          probabilities[String(cls) as "1" | "2" | "3" | "4"] ?? 0;
        const pct = p * 100;
        const isPredicted = cls === predicted;
        const { bg, fg } = NOVA_VAR[cls];
        return (
          <li key={cls} className="grid grid-cols-[1fr_auto] gap-x-3 gap-y-1">
            <span
              className={`text-sm ${isPredicted ? "font-medium text-foreground" : "text-muted-foreground"}`}
            >
              {NOVA_LABELS[cls]}
              {isPredicted && (
                <span className="ml-2 rounded-full border border-foreground/25 px-1.5 py-px text-[10px] font-semibold uppercase tracking-wide">
                  our call
                </span>
              )}
            </span>
            <span
              className={`text-sm tabular-nums ${isPredicted ? "font-medium text-foreground" : "text-muted-foreground"}`}
            >
              {pct.toFixed(1)}%
            </span>
            <div
              className="col-span-2 h-2.5 overflow-hidden rounded-full bg-background"
              role="presentation"
            >
              <div
                className="h-full rounded-full transition-[width] duration-500"
                style={{
                  width: `${Math.max(pct, 1.5)}%`,
                  backgroundColor: isPredicted ? bg : "var(--chart-3)",
                  color: fg,
                }}
                title={`${NOVA_LABELS[cls]}: ${pct.toFixed(1)}%`}
              />
            </div>
          </li>
        );
      })}
    </ul>
  );
}
