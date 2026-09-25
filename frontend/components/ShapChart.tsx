"use client";

// PRD 7.5 — "Why" section: top-5 SHAP contributions as a bar chart
// (recharts), sign = toward/away from the predicted verdict, humanized
// labels (num__additives_n → "Number of additives", text__truncatedsvd9 →
// "Ingredient wording pattern #9"). Real nutrients/frequencies/SVD —
// never PCA, per Hard rule (SHAP must explain real features). On-screen
// copy stays layman (clarify pass 25 Sep 2026); SHAP named once in the
// footnote for the course audience.

import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ShapFeature } from "@/types/predict";
import { humanizeFeature } from "@/lib/shap-labels";

interface ShapChartProps {
  features: ShapFeature[];
}

interface Row {
  label: string;
  value: number;
}

const fmt = (v: number) =>
  `${v >= 0 ? "+" : ""}${v >= 0.005 || v <= -0.005 ? v.toFixed(3) : v.toFixed(4)}`;

/** Narrow screens get a slimmer label gutter so the bars keep width. */
function useNarrowViewport(): boolean {
  const [narrow, setNarrow] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia("(max-width: 640px)");
    const update = () => setNarrow(mq.matches);
    update();
    mq.addEventListener("change", update);
    return () => mq.removeEventListener("change", update);
  }, []);
  return narrow;
}

export default function ShapChart({ features }: ShapChartProps) {
  const narrow = useNarrowViewport();

  if (!features.length) {
    return (
      <p className="text-sm text-muted-foreground">
        No reasons to show for this scan.
      </p>
    );
  }

  // SHAP bars read better bottom-up: biggest push last so it sits on top.
  const data: Row[] = features
    .map((f) => ({ label: humanizeFeature(f.feature), value: f.shap_value }))
    .slice()
    .reverse();

  return (
    <figure className="flex flex-col gap-2">
      <div
        className="h-[200px] w-full sm:h-[220px]"
        role="img"
        aria-label="The strongest reasons for this verdict, bar chart"
      >
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            layout="vertical"
            margin={{ top: 4, right: 34, bottom: 4, left: 4 }}
            barCategoryGap="24%"
          >
            <CartesianGrid horizontal={false} stroke="var(--border)" />
            <XAxis
              type="number"
              tick={{ fontSize: narrow ? 10 : 11, fill: "var(--muted-foreground)" }}
              tickFormatter={fmt}
            />
            <YAxis
              type="category"
              dataKey="label"
              width={narrow ? 104 : 150}
              tick={{ fontSize: narrow ? 10 : 11, fill: "var(--foreground)" }}
            />
            <ReferenceLine x={0} stroke="var(--foreground)" strokeOpacity={0.4} />
            <Tooltip
              cursor={{ fill: "var(--muted)" }}
              formatter={(value) => [fmt(Number(value)), "push"]}
              contentStyle={{
                borderRadius: 8,
                border: "1px solid var(--border)",
                background: "var(--popover)",
                color: "var(--popover-foreground)",
                fontSize: 12,
              }}
            />
            <Bar
              dataKey="value"
              // toward the verdict = brand honey; away = ink blue.
              // NEVER green/amber/red — those are nutrient chips only.
              radius={[0, 4, 4, 0]}
              isAnimationActive={false}
            >
              {data.map((row) => (
                <Cell
                  key={row.label}
                  fill={row.value >= 0 ? "var(--chart-1)" : "var(--chart-2)"}
                  fillOpacity={0.9}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <figcaption className="text-xs text-muted-foreground">
        Honey bars push toward this verdict, blue bars push away —
        Chewsy&rsquo;s own reading of the label. (Shown with SHAP, the
        standard way to explain a decision.)
      </figcaption>
    </figure>
  );
}
