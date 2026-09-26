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
//
// Mobile geometry (bugfix 26 Sep 2026 — "not centered / not scanning on
// phone"): html5-qrcode's foreverScan extracts the scan strip with
// `widthRatio = videoWidth / clientWidth`, which is only correct when the
// WHOLE native frame maps 1:1 onto the video's CSS box. Forcing the video
// to fill the container (`h-full object-cover`) center-crops 16:9 phone
// cameras into the 4:3 box → decoder reads a shifted/squashed region ≠
// what's on screen. Its viewfinder overlay is also sized from the VIDEO's
// client box but anchored to the viewport div, so the two only line up
// when that div hugs the video. Fix: viewport div in-flow (not
// `absolute inset-0`), video `w-full! h-auto` (native aspect, entire
// frame visible → ratio math exact), container drops its fixed 4:3 once
// scanning so viewport == video == scan region, qrbox scales with the
// viewfinder, and while scanning on phones (<md) the component goes
// fixed fullscreen so only the camera box is on screen.

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

  // Fullscreen camera on phones (<md): while scanning the whole component
  // becomes a fixed panel so only the camera box is visible; body scroll
  // locked so the page behind can't swipe under it. md+ keeps the in-card
  // layout (desktop demo verified that way).
  useEffect(() => {
    if (!scanning) return;
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previous;
    };
  }, [scanning]);

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
          // scales with the actual viewfinder (lib minimum is 50px);
          // a fixed 240×160 both mis-fits odd phone aspects and can
          // exceed short viewfinders, which makes start() throw
          qrbox: (w, h) => ({
            width: Math.min(Math.max(Math.round(w * 0.8), 100), 320),
            height: Math.min(Math.max(Math.round(h * 0.6), 60), 220),
          }),
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
    <div
      className={
        scanning
          ? "flex flex-col gap-3 max-md:fixed max-md:inset-0 max-md:z-[100] max-md:justify-center max-md:bg-background"
          : "flex flex-col gap-3"
      }
    >
      {/* Box hugs the video while scanning (see header note: viewport div
          must equal the video box or the lib's viewfinder/scan region
          drift apart); 4:3 only before start, when there's no video yet. */}
      <div
        className={`relative overflow-hidden rounded-xl border border-border bg-foreground/95 ${
          scanning ? "max-md:rounded-none max-md:border-0" : "aspect-[4/3]"
        }`}
      >
        {/* html5-qrcode injects a <video> here; empty until start().
            In-flow (not absolute) so the box and this div hug the video's
            native-aspect height — the lib anchors its scan-region overlay
            to THIS element sized from the video element. */}
        <div
          id={containerId}
          className="relative w-full [&_video]:block [&_video]:h-auto [&_video]:w-full!"
        />
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
                className="inline-flex min-h-11 items-center underline underline-offset-2 text-foreground"
                onClick={startScanner}
              >
                Try camera again
              </button>
            )}
          </AlertDescription>
        </Alert>
      )}

      <p
        className={`flex items-center gap-1.5 text-xs text-muted-foreground ${
          scanning ? "max-md:hidden" : ""
        }`}
      >
        <Keyboard className="size-3.5" aria-hidden="true" />
        {FORMAT_NOTE} Lighting or focus failing? Type it below.
      </p>
    </div>
  );
}
