"use client";

// PRD 7.2 — browser camera + client-side barcode decoding via html5-qrcode.
//
// Mechanism (recorded for the PRD): decoding reads BAR WIDTHS, not printed
// digits — it is not OCR. EAN-13's 13th digit is a checksum recomputed by
// the decoder, so a decode either succeeds correctly or emits nothing — no
// invalid barcode can reach the API, and the existing `^\d{6,14}$` backend
// validator is the only validation needed. Only the decoded digit string is
// sent over HTTP; no image leaves the browser.
//
// Why html5-qrcode and not the native BarcodeDetector API: Safari/iOS ships
// BarcodeDetector disabled by default, Firefox doesn't support it at all,
// and Chrome desktop only enables it on macOS/ChromeOS — html5-qrcode works
// everywhere a demo might be run (PRD 7.2).
//
// Config per PRD: EAN_13 / UPC_A / EAN_8 only (scanning every format every
// frame is measurably slower; these three cover Indian, European and US
// packaging), rear camera via facingMode: 'environment', and the stream is
// stopped on first successful decode — otherwise it keeps firing and
// double-submits.

import { useCallback, useEffect, useRef, useState } from "react";
import { Camera, CameraOff, Keyboard, ShieldAlert, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

type CameraError =
  | { kind: "insecure"; message: string }
  | { kind: "permission"; message: string }
  | { kind: "nodevice"; message: string }
  | { kind: "unknown"; message: string };

interface BarcodeScannerProps {
  /** Fires once per successful decode — camera is stopped first. */
  onDecode: (barcode: string) => void;
  /** True while a prediction is in flight — blocks re-triggering. */
  busy: boolean;
}

const FORMAT_NOTE =
  "Decodes EAN-13 / UPC-A / EAN-8 by bar widths (checksum-validated).";

export default function BarcodeScanner({ onDecode, busy }: BarcodeScannerProps) {
  const [scanning, setScanning] = useState(false);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<CameraError | null>(null);
  const scannerRef = useRef<{
    stop: () => Promise<void>;
    clear: () => void;
  } | null>(null);
  const containerId = "chewsy-scanner-viewport";

  const stopScanner = useCallback(async () => {
    const scanner = scannerRef.current;
    scannerRef.current = null;
    if (scanner) {
      try {
        await scanner.stop();
        scanner.clear();
      } catch {
        // already stopped / element removed — nothing to release
      }
    }
    setScanning(false);
  }, []);

  // Release the camera on unmount (navigation, result screen swap).
  useEffect(() => {
    return () => {
      const scanner = scannerRef.current;
      scannerRef.current = null;
      if (scanner) {
        scanner.stop().catch(() => {}).then(() => scanner.clear());
      }
    };
  }, []);

  const startScanner = useCallback(async () => {
    setError(null);
    setStarting(true);
    try {
      // Secure-context gate (PRD 7.10): plain http:// away from localhost
      // has no mediaDevices at all — name it instead of showing a dead view.
      if (
        typeof navigator === "undefined" ||
        !navigator.mediaDevices ||
        typeof navigator.mediaDevices.getUserMedia !== "function"
      ) {
        setError({
          kind: "insecure",
          message:
            "Camera requires HTTPS — your browser blocks cameras on plain HTTP. Type the barcode below instead.",
        });
        return;
      }

      const { Html5Qrcode, Html5QrcodeSupportedFormats } = await import(
        "html5-qrcode"
      );
      const scanner = new Html5Qrcode(containerId, {
        formatsToSupport: [
          Html5QrcodeSupportedFormats.EAN_13,
          Html5QrcodeSupportedFormats.UPC_A,
          Html5QrcodeSupportedFormats.EAN_8,
        ],
        verbose: false,
      });
      scannerRef.current = scanner;

      await scanner.start(
        { facingMode: "environment" },
        {
          fps: 10,
          qrbox: { width: 240, height: 160 },
        },
        (decodedText) => {
          // Stop on first decode — the stream would otherwise keep firing
          // and double-submit (PRD 7.2 config specifics).
          void stopScanner().then(() => onDecode(decodedText.trim()));
        },
        () => {
          // per-frame miss — expected, not an error
        }
      );
      setScanning(true);
    } catch (err) {
      scannerRef.current = null;
      const name = err instanceof Error ? err.name : "";
      if (name === "NotAllowedError" || name === "PermissionDeniedError") {
        setError({
          kind: "permission",
          message:
            "Camera permission denied — allow it in your browser settings, or type the barcode below.",
        });
      } else if (
        name === "NotFoundError" ||
        name === "OverconstrainedError" ||
        name === "DevicesNotFoundError"
      ) {
        setError({
          kind: "nodevice",
          message: "No camera found on this device — type the barcode below.",
        });
      } else {
        setError({
          kind: "unknown",
          message: "Camera could not start — type the barcode below instead.",
        });
      }
    } finally {
      setStarting(false);
    }
  }, [onDecode, stopScanner]);

  return (
    <div className="flex flex-col gap-3">
      <div className="relative overflow-hidden rounded-xl border border-border bg-foreground/95 aspect-[4/3]">
        {/* html5-qrcode injects a <video> here; empty until start(). */}
        <div id={containerId} className="absolute inset-0 [&_video]:h-full [&_video]:w-full [&_video]:object-cover" />
        {!scanning && !starting && (
          <div className="absolute inset-0 grid place-content-center justify-items-center gap-3 p-6 text-center">
            <Camera className="size-8 text-primary" aria-hidden="true" />
            <p className="text-sm text-muted-foreground/90 max-w-60">
              Point the camera at a grocery barcode.
            </p>
            <Button onClick={startScanner} disabled={busy}>
              <Camera className="size-4" /> Start camera
            </Button>
          </div>
        )}
        {starting && (
          <div className="absolute inset-0 grid place-content-center justify-items-center gap-2 bg-background/80">
            <Loader2 className="size-6 animate-spin text-primary" aria-hidden="true" />
            <p className="text-sm text-muted-foreground">Starting camera…</p>
          </div>
        )}
        {scanning && (
          <div className="absolute inset-x-0 bottom-0 flex items-center justify-between gap-2 bg-background/85 px-3 py-2">
            <span className="text-xs text-muted-foreground">Scanning…</span>
            <Button size="sm" variant="secondary" onClick={stopScanner}>
              Stop
            </Button>
          </div>
        )}
      </div>

      {error && (
        <Alert variant={error.kind === "insecure" ? "default" : "destructive"}>
          {error.kind === "insecure" ? (
            <ShieldAlert />
          ) : error.kind === "permission" ? (
            <ShieldAlert />
          ) : (
            <CameraOff />
          )}
          <AlertTitle>
            {error.kind === "insecure"
              ? "Camera needs a secure page"
              : error.kind === "permission"
                ? "Camera blocked"
                : "Camera unavailable"}
          </AlertTitle>
          <AlertDescription>
            {error.message}{" "}
            {error.kind !== "insecure" && (
              <button
                type="button"
                className="underline underline-offset-2 text-foreground"
                onClick={startScanner}
              >
                Try camera again
              </button>
            )}
          </AlertDescription>
        </Alert>
      )}

      <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
        <Keyboard className="size-3.5" aria-hidden="true" />
        {FORMAT_NOTE} Lighting or focus failing? Type it below.
      </p>
    </div>
  );
}
