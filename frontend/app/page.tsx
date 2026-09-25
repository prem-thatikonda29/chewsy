"use client";

// PRD 7.4 / 7.9 / 7.10 — scan flow state machine. The ONLY data path is
// lib/api.ts → FastAPI POST /predict over HTTP: this UI never computes a
// score, band, or headline (one source of truth).

import { useCallback, useState } from "react";
import { Loader2, RotateCw, Search, TriangleAlert } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import BarcodeScanner from "@/components/BarcodeScanner";
import ResultCard from "@/components/ResultCard";
import { SectionHead, Well } from "@/components/Well";
import { BARCODE_PATTERN, ScanError, predict, type ScanErrorKind } from "@/lib/api";
import type { PredictResponse } from "@/types/predict";

type Screen =
  | { s: "scan" }
  | { s: "loading"; barcode: string }
  | { s: "result"; data: PredictResponse }
  | { s: "error"; barcode: string; kind: ScanErrorKind; message: string };

interface HistoryEntry {
  barcode: string;
  name: string;
  nova: number;
}

const HISTORY_LIMIT = 10;

const ERROR_COPY: Record<ScanErrorKind, { title: string; action: "retry" | "rescan" }> = {
  not_found: { title: "Product not found", action: "rescan" },
  unavailable: { title: "Couldn't reach the food database", action: "retry" },
  invalid: { title: "Not a valid barcode", action: "rescan" },
  unknown: { title: "Something went wrong", action: "retry" },
};

export default function Home() {
  const [screen, setScreen] = useState<Screen>({ s: "scan" });
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [manualValue, setManualValue] = useState("");
  const [manualError, setManualError] = useState<string | null>(null);

  const runScan = useCallback(async (rawBarcode: string) => {
    const barcode = rawBarcode.trim();
    if (!BARCODE_PATTERN.test(barcode)) {
      setManualError("Barcode must be 6–14 digits.");
      setScreen({ s: "scan" });
      return;
    }
    setManualError(null);
    setManualValue("");
    setScreen({ s: "loading", barcode });
    try {
      const data = await predict(barcode);
      setHistory((prev) => {
        const next = [
          {
            barcode: data.barcode,
            name: data.product_name || data.barcode,
            nova: data.predicted_nova,
          },
          ...prev.filter((h) => h.barcode !== data.barcode),
        ];
        return next.slice(0, HISTORY_LIMIT);
      });
      setScreen({ s: "result", data });
    } catch (err) {
      if (err instanceof ScanError) {
        setScreen({
          s: "error",
          barcode,
          kind: err.kind,
          message: err.message,
        });
      } else {
        setScreen({
          s: "error",
          barcode,
          kind: "unknown",
          message: "Unexpected error — try again.",
        });
      }
    }
  }, []);

  const submitManual = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      void runScan(manualValue);
    },
    [manualValue, runScan]
  );

  if (screen.s === "result") {
    return (
      <div className="mx-auto w-full max-w-3xl px-4 py-6">
        <ResultCard
          data={screen.data}
          onScanAnother={() => setScreen({ s: "scan" })}
        />
      </div>
    );
  }

  if (screen.s === "loading") {
    return (
      <div
        className="mx-auto w-full max-w-3xl px-4 py-6"
        aria-busy="true"
        role="status"
        aria-live="polite"
      >
        <Card className="gap-0">
          <CardContent className="flex flex-col gap-5 px-4 py-5 sm:px-6">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="size-4 animate-spin text-primary" aria-hidden="true" />
              Looking up <span className="font-mono">{screen.barcode}</span> —
              fetching the label, then working out how it&rsquo;s made…
            </div>
            <div className="flex items-start gap-4">
              <Skeleton className="size-20 shrink-0 rounded-lg sm:size-24" />
              <div className="flex flex-1 flex-col gap-2">
                <Skeleton className="h-5 w-2/3" />
                <Skeleton className="h-4 w-1/3" />
                <Skeleton className="h-6 w-1/2" />
              </div>
            </div>
            <Skeleton className="h-24 w-full" />
            <Skeleton className="h-32 w-full" />
          </CardContent>
        </Card>
      </div>
    );
  }

  if (screen.s === "error") {
    const copy = ERROR_COPY[screen.kind];
    return (
      <div className="mx-auto w-full max-w-3xl px-4 py-6">
        <Card className="gap-0">
          <CardContent className="flex flex-col gap-5 px-4 py-5 sm:px-6">
            <Alert>
              <TriangleAlert />
              <AlertTitle>{copy.title}</AlertTitle>
              <AlertDescription>
                <span className="font-mono text-xs">{screen.barcode}</span>
                {" — "}
                {screen.message}
              </AlertDescription>
            </Alert>
            <div className="flex flex-wrap gap-2">
              {copy.action === "retry" && (
                <Button onClick={() => void runScan(screen.barcode)}>
                  <RotateCw className="size-4" /> Retry
                </Button>
              )}
              <Button
                variant={copy.action === "retry" ? "outline" : "default"}
                onClick={() => setScreen({ s: "scan" })}
              >
                Scan another
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  // --- scan screen ---
  return (
    <div className="mx-auto w-full max-w-3xl px-4 py-6 flex flex-col gap-5">
      <div className="flex flex-col gap-1">
        <h1 className="text-xl font-semibold tracking-tight">
          Scan a barcode
        </h1>
        <p className="text-sm text-muted-foreground">
          One scan answers two questions: how it&rsquo;s made, and what&rsquo;s
          in it.
        </p>
      </div>

      <Card className="gap-0">
        <CardContent className="flex flex-col gap-5 px-4 py-5 sm:px-6">
          <BarcodeScanner onDecode={(code) => void runScan(code)} busy={false} />

          {/* 7.3 — manual entry sits alongside the camera view: every real
              scanner app has one, for lighting/focus failures mid-demo.
              Stacks full-width on phones so both targets stay >=44px. */}
          <form onSubmit={submitManual} className="flex flex-col gap-2">
            <label htmlFor="manual-barcode" className="text-sm font-medium">
              Enter barcode manually
            </label>
            <div className="flex flex-col gap-2 sm:flex-row">
              <Input
                id="manual-barcode"
                inputMode="numeric"
                autoComplete="off"
                placeholder="e.g. 3017620422003"
                value={manualValue}
                onChange={(e) => {
                  setManualValue(e.target.value.replace(/\D/g, ""));
                  setManualError(null);
                }}
                aria-invalid={manualError ? true : undefined}
                className="font-mono"
              />
              <Button type="submit" disabled={!manualValue} className="sm:w-auto">
                <Search className="size-4" /> Look up
              </Button>
            </div>
            {manualError && (
              <p className="text-xs text-destructive" role="alert">
                {manualError}
              </p>
            )}
          </form>
        </CardContent>
      </Card>

      {/* 7.9 — session scan history, last 10 (same well recipe as the
          result card's sections). */}
      {history.length > 0 && (
        <Well as="section" className="flex flex-col gap-3" aria-label="Recent scans">
          <SectionHead title="Recent scans" caption="this session · tap to re-scan" />
          <div className="flex flex-wrap gap-2">
            {history.map((entry) => (
              <button
                key={entry.barcode}
                type="button"
                onClick={() => void runScan(entry.barcode)}
                className="inline-flex max-w-64 items-center gap-1.5 rounded-full border border-border bg-card px-3 py-1.5 text-xs transition-colors hover:bg-accent"
                title={`Re-scan ${entry.barcode}`}
              >
                <span className="truncate">{entry.name}</span>
                <span
                  className="shrink-0 rounded-full px-1.5 py-px font-semibold"
                  style={{
                    backgroundColor: `var(--nova-${entry.nova})`,
                    color: `var(--nova-${entry.nova}-fg)`,
                  }}
                >
                  {entry.nova}
                </span>
              </button>
            ))}
          </div>
        </Well>
      )}
    </div>
  );
}
