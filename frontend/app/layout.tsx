import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Chewsy — scan a barcode, see what's in it",
  description:
    "Scan a grocery barcode: Chewsy works out how the food is made (NOVA 1–4) from its ingredients and shows the nutrient traffic lights — its own verdict.",
};

// Required over plain HTTP away from localhost: getUserMedia (the camera)
// only works in a secure context — Stage 11's known limitation, PRD 11.3.
// viewport-fit=cover lets the safe-area insets in globals.css keep
// content clear of notches / the home indicator.
export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">
        <header className="border-b border-border/70 bg-card/80 backdrop-blur supports-[backdrop-filter]:bg-card/60 sticky top-0 z-10">
          <div className="mx-auto w-full max-w-3xl px-4 py-3 flex items-center justify-between gap-3">
            <span className="flex items-center gap-2">
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth={2.2}
                strokeLinecap="round"
                className="size-5 text-primary"
                aria-hidden="true"
              >
                <path d="M3 7V5a2 2 0 0 1 2-2h2" />
                <path d="M17 3h2a2 2 0 0 1 2 2v2" />
                <path d="M21 17v2a2 2 0 0 1-2 2h-2" />
                <path d="M7 21H5a2 2 0 0 1-2-2v-2" />
                <path d="M7 8v8" />
                <path d="M11 8v8" />
                <path d="M15 8v8" />
              </svg>
              <span className="text-base font-semibold tracking-tight">
                Chewsy
              </span>
            </span>
            <span className="text-xs text-muted-foreground">
              how it&rsquo;s made · what&rsquo;s in it
            </span>
          </div>
        </header>
        <main className="flex-1">{children}</main>
        <footer className="border-t border-border/70 py-4">
          <p className="mx-auto w-full max-w-3xl px-4 text-xs text-muted-foreground">
            Chewsy works out how each product is made — a description of
            processing, not a health score. See the explainer on every
            result.
          </p>
        </footer>
      </body>
    </html>
  );
}
