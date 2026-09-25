# Chewsy frontend

Next.js (App Router, TypeScript, Tailwind 4, shadcn/ui) scanner UI for the
Chewsy barcode → NOVA classifier (Stage 7 of `Chewsy_PRD_Roadmap.md`).

## Run

```bash
npm install
npm run dev        # http://localhost:3000
```

Requires the FastAPI backend on port 8000 (`uvicorn app.main:app` from the
repo root). The API origin is the `NEXT_PUBLIC_API_URL` env var
(`.env.local` for local dev) — it is **inlined at build time**, so Docker/
EC2 builds must pass it during `npm run build`.

## Scripts

- `npm run dev` — dev server (port 3000)
- `npm run build` — production build
- `npm run typecheck` — `tsc --noEmit` (Next 16 removed `next lint`'s
  typecheck role; this is PRD 7.11's gate)
- `npm run lint` — eslint

## Notes

- The UI only ever POSTs `{barcode}` to `/predict` over HTTP — no scoring
  logic or model lives here (`lib/api.ts` is the single data path).
- Camera decoding is client-side (`html5-qrcode`, EAN-13/UPC-A/EAN-8);
  only the decoded digit string is sent to the backend. Camera requires a
  secure context (localhost or HTTPS).
