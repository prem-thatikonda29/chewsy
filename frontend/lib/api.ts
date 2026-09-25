// PRD 7.4/7.6/7.10 — the ONE way this UI talks to the backend.
// HTTP POST to FastAPI /predict; no scoring logic, no model, no band
// math ever lives here (one source of truth).
//
// NEXT_PUBLIC_ vars are inlined at build time (Next.js docs) — Docker/
// EC2 builds pass their own value at `next build`; local dev falls back
// to the FastAPI default port.

import type { PredictResponse } from "@/types/predict";

const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(
  /\/$/,
  ""
);

export type ScanErrorKind = "not_found" | "unavailable" | "invalid" | "unknown";

/** Typed failure so the UI can name the real cause (PRD 7.10). */
export class ScanError extends Error {
  constructor(
    readonly kind: ScanErrorKind,
    message: string,
    readonly status?: number
  ) {
    super(message);
    this.name = "ScanError";
  }
}

export const BARCODE_PATTERN = /^\d{6,14}$/;

export async function predict(barcode: string): Promise<PredictResponse> {
  if (!BARCODE_PATTERN.test(barcode)) {
    throw new ScanError("invalid", "Barcode must be 6-14 digits.");
  }

  let res: Response;
  try {
    res = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ barcode }),
    });
  } catch {
    // network-level failure (API down, offline) — distinct from a 503
    throw new ScanError("unavailable", "API unreachable — check the backend is running.");
  }

  if (res.status === 404) {
    throw new ScanError("not_found", "Product not found in OFF.", 404);
  }
  if (res.status === 422) {
    throw new ScanError("invalid", "Invalid barcode format.", 422);
  }
  if (res.status === 503 || res.status === 429 || res.status >= 500) {
    throw new ScanError(
      "unavailable",
      "OFF unreachable — retry in a moment.",
      res.status
    );
  }
  if (!res.ok) {
    throw new ScanError("unknown", `Unexpected error (HTTP ${res.status}).`, res.status);
  }

  return (await res.json()) as PredictResponse;
}
