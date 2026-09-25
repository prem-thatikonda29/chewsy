# Product

## Register

product

## Users

- **Primary:** course judges (FE & MLOps mini-project) watching a live
  demo — they need the verdict, the evidence, and the honesty of the
  framing to be legible in seconds, on whatever screen they're given.
- **Scenario user:** a grocery-shopper-style scan (Yuka-style): phone
  camera on a real package, wanting to know *how it's made* and *what's
  in it* without being told the food is "good" or "bad".

## Product Purpose

Chewsy answers two separate questions about one product: **how it's
made** (NOVA 1–4, model-computed from ingredients) and **what's in it**
(NHS traffic-light bands computed locally from published thresholds).
The model runs on every scan; the UI must make the model's work visible
(SHAP, probabilities) and never launder OFF's labels. Success: a viewer
understands both axes, trusts the model because its reasoning is shown,
and leaves knowing NOVA describes formulation, not healthiness.

## Brand Personality

Honest · Calm · Credible.

Plain-spoken and evidence-first. No health theatrics, no scare color,
no superlatives. The voice states what was measured and what wasn't.

## Anti-references

- Yuka-style single red/green "health score" verdict — we refuse one
  number by design (two axes, kept separate).
- Green→red NOVA badge (binding PRD rule: processing level is neutral,
  not a judgment).
- Calorie-diet-app aesthetics (MyFitnessPal-style dashboards, streaks).
- Any Nutri-Score display — OFF's or ours (Hard rules 11 + 12).
- Slop tells: gradient text, health-guru copy, decorative noise.

## Design Principles

1. **NOVA is a descriptor, never a verdict** (binding, PRD §1) — the UI
   never presents a processing class as good/bad; the descriptor line
   appears on the card itself.
2. **Two axes, visually separate** — processing level uses a neutral
   light→dark ramp; green/amber/red is reserved exclusively for nutrient
   chips where it matches published FSA meaning.
3. **Show the model's work** — probabilities and SHAP are first-class,
   never a bare number; ambiguity (Borderline) is stated, not hidden.
4. **Honest states** — null renders as "—", missing nutrition data says
   so, bands are never guessed, errors name the real cause (404 vs 503).
5. **One source of truth** — the UI calls the API over HTTP and never
   recomputes or duplicates scoring.

## Accessibility & Inclusion

WCAG 2.1 AA: body-text contrast ≥ 4.5:1; nutrient chips carry
text/icon labels, not color alone (color-blind safe); keyboard-operable
manual barcode entry; camera failure states (insecure context,
permission denied, no device) degrade to manual entry with explicit
messages; `prefers-reduced-motion` respected for any animation.
