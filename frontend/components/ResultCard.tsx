"use client";

// PRD 7.5 — the single scrollable result card (structure approved
// 25 Sep 2026, two-axis reframe). Top → bottom: hero (headline largest,
// confidence chip) · two-axis row · probability distribution · SHAP why ·
// facts panel · NOVA explainer (always visible) · footer microcopy.
//
// Binding colour rule: the NOVA badge is a NEUTRAL light→dark ramp —
// never green→red, because red reads as "bad", the exact conflation this
// tool fixes. Green/amber/red appears ONLY on nutrient traffic-light
// chips, where it matches the published FSA meaning.

import Image from "next/image";
import {
  CircleAlert,
  CircleCheck,
  HelpCircle,
  ImageOff,
  RefreshCw,
  TriangleAlert,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import ProbabilityBars from "@/components/ProbabilityBars";
import ShapChart from "@/components/ShapChart";
import { SectionHead, Well } from "@/components/Well";
import {
  NUTRITION_ROWS,
  NOVA_LABELS,
  TRAFFIC_KEYS,
  type NovaClass,
  type PredictResponse,
  type TrafficKey,
} from "@/types/predict";

interface ResultCardProps {
  data: PredictResponse;
  onScanAnother: () => void;
}

const TRAFFIC_META: Record<
  TrafficKey,
  { label: string; fg: string; bg: string; Icon: typeof CircleCheck }
> = {
  sugars: { label: "Sugars", fg: "", bg: "", Icon: CircleCheck },
  fat: { label: "Fat", fg: "", bg: "", Icon: CircleCheck },
  saturated_fat: { label: "Sat fat", fg: "", bg: "", Icon: CircleCheck },
  salt: { label: "Salt", fg: "", bg: "", Icon: CircleCheck },
};

const BAND_STYLE: Record<
  string,
  { bg: string; fg: string; Icon: typeof CircleCheck; word: string }
> = {
  low: { bg: "var(--fsa-low)", fg: "var(--fsa-low-fg)", Icon: CircleCheck, word: "low" },
  medium: { bg: "var(--fsa-medium)", fg: "var(--fsa-medium-fg)", Icon: TriangleAlert, word: "medium" },
  high: { bg: "var(--fsa-high)", fg: "var(--fsa-high-fg)", Icon: CircleAlert, word: "high" },
  missing: { bg: "var(--fsa-none)", fg: "var(--fsa-none-fg)", Icon: HelpCircle, word: "not published" },
};

const NOVA_STYLE: Record<NovaClass, { bg: string; fg: string }> = {
  1: { bg: "var(--nova-1)", fg: "var(--nova-1-fg)" },
  2: { bg: "var(--nova-2)", fg: "var(--nova-2-fg)" },
  3: { bg: "var(--nova-3)", fg: "var(--nova-3-fg)" },
  4: { bg: "var(--nova-4)", fg: "var(--nova-4-fg)" },
};

function confidenceTier(confidence: number, probabilities: Record<string, number>): {
  word: string;
  variant: "solid" | "outline" | "dashed";
} {
  const sorted = Object.values(probabilities).sort((a, b) => b - a);
  const gap = ((sorted[0] ?? 0) - (sorted[1] ?? 0)) * 100;
  if (confidence < 0.6 || gap < 15) return { word: "Borderline", variant: "outline" };
  if (confidence >= 0.8) return { word: "Confident", variant: "solid" };
  return { word: "Fairly sure", variant: "outline" };
}

function NutrientChip({ band }: { band: string | null }) {
  const style = BAND_STYLE[band ?? "missing"] ?? BAND_STYLE.missing;
  const { Icon } = style;
  return (
    <span
      className="inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-xs font-medium"
      style={{ backgroundColor: style.bg, color: style.fg }}
    >
      <Icon className="size-3.5" aria-hidden="true" />
      {style.word}
    </span>
  );
}

export default function ResultCard({ data, onScanAnother }: ResultCardProps) {
  const tier = confidenceTier(data.confidence, data.class_probabilities);
  const nova = data.predicted_nova as NovaClass;
  const novaStyle = NOVA_STYLE[nova] ?? NOVA_STYLE[4];

  return (
    <Card className="gap-0">
      <CardContent className="flex flex-col gap-5 px-4 py-5 sm:px-6">
        {/* 1 — Hero: headline is the LARGEST text (combined two-axis
            sentence leads, not the badge). */}
        <section className="flex flex-col gap-3" aria-label="Verdict">
          <div className="flex items-start gap-4">
            <div className="relative size-20 shrink-0 overflow-hidden rounded-lg border border-border bg-muted sm:size-24">
              {data.image_url ? (
                <Image
                  src={data.image_url}
                  alt={data.product_name || "Product image"}
                  fill
                  unoptimized
                  className="object-contain"
                  onError={(e) => {
                    (e.currentTarget as HTMLImageElement).style.display = "none";
                  }}
                />
              ) : (
                <div className="absolute inset-0 grid place-content-center justify-items-center gap-1 text-muted-foreground">
                  <ImageOff className="size-5" aria-hidden="true" />
                </div>
              )}
            </div>
            <div className="flex min-w-0 flex-col gap-1.5">
              <h1 className="truncate text-lg font-semibold leading-snug">
                {data.product_name || "Unnamed product"}
              </h1>
              <p className="font-mono text-xs text-muted-foreground">
                {data.barcode}
              </p>
              <span
                className={
                  "inline-flex w-fit items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-semibold " +
                  (tier.variant === "solid"
                    ? "bg-foreground text-background"
                    : "border border-foreground/40 text-foreground")
                }
                title={`Model confidence ${(data.confidence * 100).toFixed(1)}%`}
              >
                <HelpCircle className="size-3.5" aria-hidden="true" />
                {tier.word} · {(data.confidence * 100).toFixed(0)}%
              </span>
            </div>
          </div>
          <p className="text-2xl font-semibold leading-tight tracking-tight sm:text-3xl">
            {data.headline}
          </p>
        </section>

        {/* 2 — Two-axis row: what's made vs what's in it, side by side. */}
        <section
          className="grid gap-4 sm:grid-cols-2"
          aria-label="Two-axis summary"
        >
          <Well className="flex flex-col gap-2">
            <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
              How it&rsquo;s made
            </p>
            <span
              className="inline-flex w-fit items-center gap-2 rounded-full px-3 py-1.5 text-sm font-semibold"
              style={{ backgroundColor: novaStyle.bg, color: novaStyle.fg }}
            >
              NOVA {nova}
              <span className="font-normal opacity-80">·</span>
              <span className="font-medium">{NOVA_LABELS[nova]}</span>
            </span>
          </Well>
          <Well className="flex flex-col gap-2">
            <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
              What&rsquo;s in it
            </p>
            <div className="flex flex-wrap gap-1.5">
              {TRAFFIC_KEYS.map((key) => (
                <span key={key} className="inline-flex items-center gap-1">
                  <span className="text-xs text-muted-foreground">
                    {TRAFFIC_META[key].label}
                  </span>
                  <NutrientChip band={data.traffic_lights[key] ?? null} />
                </span>
              ))}
            </div>
          </Well>
        </section>

        {/* 3 — Probability distribution: ambiguity made visible. */}
        <Well as="section" className="flex flex-col gap-3" aria-label="Class probabilities">
          <SectionHead title="How sure is the model?" caption="4-class distribution" />
          <ProbabilityBars
            probabilities={data.class_probabilities}
            predicted={nova}
          />
        </Well>

        {/* 4 — Why: SHAP top features. */}
        <Well as="section" className="flex flex-col gap-3" aria-label="Why this prediction">
          <SectionHead
            title="Why — what the model read"
            caption={`top 5 signals for NOVA ${nova}`}
          />
          <ShapChart features={data.shap_top_features} />
        </Well>

        {/* 5 — Facts panel: per-100g grid (null → "—"), counts, ingredients. */}
        <Well as="section" className="flex flex-col gap-3" aria-label="Nutrition facts">
          <SectionHead
            title="Facts per 100g"
            caption={`${data.additives_n} additive${data.additives_n === 1 ? "" : "s"} · ${data.ingredients_n} ingredient${data.ingredients_n === 1 ? "" : "s"}`}
          />
          <dl className="grid grid-cols-1 gap-x-6 gap-y-1 sm:grid-cols-2">
            {NUTRITION_ROWS.map(({ key, label, unit }) => {
              const value = data.nutrition_100g[key];
              // Stage 6.7 positives: fibre/protein context chips — rendered
              // as muted OUTLINE chips so they never read as traffic lights
              // (the FSA scheme has no official thresholds for them).
              const band =
                key === "fiber" || key === "proteins"
                  ? data.positives[key]
                  : null;
              return (
                <div
                  key={key}
                  className="flex items-baseline justify-between gap-3 border-b border-border/60 py-1 text-sm last:border-b-0"
                >
                  <dt className="text-muted-foreground">{label}</dt>
                  <dd className="flex items-baseline gap-2 font-medium tabular-nums">
                    {value === null || value === undefined ? (
                      <span aria-label="not published">—</span>
                    ) : (
                      <>
                        {value}
                        <span className="text-xs font-normal text-muted-foreground">
                          {unit}
                        </span>
                      </>
                    )}
                    {band && (
                      <span className="rounded-full border border-border px-1.5 py-px text-[10px] font-medium text-muted-foreground">
                        {band}
                      </span>
                    )}
                  </dd>
                </div>
              );
            })}
          </dl>
          {(data.positives.fiber || data.positives.proteins) && (
            <p className="text-xs text-muted-foreground">
              Fibre &amp; protein tags are Chewsy&rsquo;s own bands — not
              official traffic lights.
            </p>
          )}
          <div className="rounded-lg bg-background/80 p-3">
            <p className="mb-1 text-xs font-medium uppercase tracking-wide text-muted-foreground">
              Ingredients
            </p>
            <p className="text-sm leading-relaxed">
              {data.ingredients_text?.trim() || (
                <span className="text-muted-foreground">—</span>
              )}
            </p>
          </div>
        </Well>

        {/* 6 — NOVA explainer: always visible, descriptor-not-verdict. */}
        <Well as="section" className="flex flex-col gap-2" aria-label="About NOVA">
          <SectionHead title="What NOVA means" caption="descriptor, not verdict" />
          <p className="text-sm leading-relaxed text-muted-foreground">
            NOVA describes how a food is made, not how nutritious it is. NOVA 1
            is unprocessed or minimally processed; NOVA 4 is ultra-processed —
            a processing level, not a health score. Read the nutrient chips
            above for that.
          </p>
        </Well>

        {/* 7 — Footer microcopy (pitch credibility) + reset. */}
        <footer className="flex flex-col items-start gap-3">
          <p className="text-xs text-muted-foreground">
            Chewsy model verdict — computed from ingredients, not copied from
            OFF.
          </p>
          <Button onClick={onScanAnother}>
            <RefreshCw className="size-4" /> Scan another
          </Button>
        </footer>
      </CardContent>
    </Card>
  );
}
