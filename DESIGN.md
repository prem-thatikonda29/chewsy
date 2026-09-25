---
name: Chewsy
description: Calm two-axis barcode scanner — how a food is made and what's in it, shown separately and honestly.
colors:
  primary: "oklch(0.49 0.105 75)"
  background: "oklch(1 0 0)"
  foreground: "oklch(0.17 0.012 75)"
  secondary: "oklch(0.968 0.008 75)"
  secondary-foreground: "oklch(0.2 0.012 75)"
  muted: "oklch(0.968 0.008 75)"
  muted-foreground: "oklch(0.475 0.012 75)"
  border: "oklch(0.922 0.008 75)"
  ring: "oklch(0.48 0.085 255)"
  fsa-low: "oklch(0.46 0.1 150)"
  fsa-medium: "oklch(0.78 0.11 75)"
  fsa-high: "oklch(0.455 0.13 27)"
  fsa-none: "oklch(0.968 0.008 75)"
  nova-1: "oklch(0.94 0.004 250)"
  nova-2: "oklch(0.85 0.005 250)"
  nova-3: "oklch(0.68 0.006 250)"
  nova-4: "oklch(0.32 0.008 250)"
  on-saturated: "oklch(0.99 0 0)"
  on-amber: "oklch(0.22 0.04 75)"
typography:
  display:
    fontFamily: "Geist, system-ui, sans-serif"
    fontSize: "30px"
    fontWeight: 600
    lineHeight: 1.15
    letterSpacing: "-0.02em"
  title:
    fontFamily: "Geist, system-ui, sans-serif"
    fontSize: "18px"
    fontWeight: 600
    lineHeight: 1.375
    letterSpacing: "-0.01em"
  body:
    fontFamily: "Geist, system-ui, sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.5
    letterSpacing: "normal"
  label:
    fontFamily: "Geist, system-ui, sans-serif"
    fontSize: "12px"
    fontWeight: 500
    lineHeight: 1
    letterSpacing: "normal"
  mono:
    fontFamily: "Geist Mono, ui-monospace, monospace"
    fontSize: "12px"
    fontWeight: 400
    lineHeight: 1.4
    letterSpacing: "normal"
rounded:
  sm: "7.2px"
  md: "9.6px"
  lg: "12px"
  xl: "16.8px"
  pill: "9999px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "16px"
  lg: "24px"
  xl: "32px"
components:
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.on-saturated}"
    rounded: "{rounded.lg}"
    height: "32px"
    padding: "0 10px"
  button-secondary:
    backgroundColor: "{colors.secondary}"
    textColor: "{colors.secondary-foreground}"
    rounded: "{rounded.lg}"
    height: "32px"
    padding: "0 10px"
  input:
    backgroundColor: "{colors.background}"
    textColor: "{colors.foreground}"
    rounded: "{rounded.lg}"
    height: "32px"
    padding: "0 10px"
  card:
    backgroundColor: "{colors.background}"
    textColor: "{colors.foreground}"
    rounded: "{rounded.xl}"
    padding: "16px"
  chip-nutrient:
    backgroundColor: "{colors.fsa-low}"
    textColor: "{colors.on-saturated}"
    rounded: "{rounded.pill}"
    padding: "4px 10px"
  chip-nova:
    backgroundColor: "{colors.nova-4}"
    textColor: "{colors.on-saturated}"
    rounded: "{rounded.pill}"
    padding: "6px 12px"
  confidence-chip:
    backgroundColor: "{colors.foreground}"
    textColor: "{colors.background}"
    rounded: "{rounded.pill}"
    padding: "4px 10px"
  nav-bar:
    backgroundColor: "{colors.background}"
    textColor: "{colors.foreground}"
    padding: "12px 16px"
---

# Design System: Chewsy

> Tokens are OKLCH throughout — this project's canonical format (Tailwind
> CSS-first, no hex conversion layer). Expect a Stitch linter warning on
> color format; that is accepted posture, not an error.

## 1. Overview

**Creative North Star: "The Two-Question Scale"**

Chewsy is a small, calm instrument for one job: put a barcode in, get two
separate answers out — *how it's made* and *what's in it*. The system
borrows its manners from the package label itself: plain, precise,
evidence-first, never shouting. Personality is **Honest · Calm ·
Credible** — "no health theatrics, no scare color, no superlatives."
Every surface is white or near-white; one saturated accent (Pressed
Honey) marks actions; semantic color appears only where it carries
published meaning. The restraint is the credibility.

The aesthetic philosophy is **calm instrument, not a wellness app**. It
explicitly rejects wellness-app gradients, guilt-red verdicts, gamified
scoring, and SaaS dashboard chrome — plus PRODUCT.md's named
anti-references: the Yuka-style single red/green "health score", the
green→red NOVA badge, calorie-diet-app aesthetics, any Nutri-Score
display, and "slop tells: gradient text, health-guru copy, decorative
noise." Depth is carried by 1px borders and tonal steps, never shadow.
Motion is state feedback only,150–250ms, reduced-motion honored.

**Key Characteristics:**
- White field, warm near-black ink, one honey accent on ≤ ~10% of any screen.
- Two axes kept visually separate: neutral grey ramp for processing,
  green/amber/red only for nutrient bands.
- Model's work is on the card: probabilities, SHAP, confidence tier —
  ambiguity stated, never hidden.
- Flat surfaces, 1px borders, 12–16.8px radii, 32px controls on fine
  pointers / 44px on touch, mono for barcodes and data.
- Honest states everywhere: null renders as "—", errors name the real cause.

## 2. Colors

A restrained palette: pure white surfaces, one food-adjacent accent, and
two strictly-scoped semantic families.

### Primary
- **Pressed Honey** (`primary`, oklch(0.49 0.105 75)): the only
  decorative-saturation color. Fires on primary buttons, the logo mark,
  loading spinner, and "toward the verdict" SHAP bars. Chroma is held at
  0.105 — ~20% below full saturation (quieter pass, 25 Sep 2026) so the
  accent reads soft — while L 0.49 keeps white-on-honey ≥ 4.5:1. It never
  describes good or bad — it only means "the action" or "the model pushed
  this way."

### Secondary
- **Bench Blue** (`ring`, oklch(0.48 0.085 255)): focus rings, the
  "pushes away from the verdict" SHAP bars, and error-path attention. The
  cool counterweight to honey — instrument, not food-label.

### Neutral
- **White Field** (`background`, oklch(1 0 0)): every page and card surface.
- **Warm Ink** (`foreground`, oklch(0.17 0.012 75)): all primary text; ≥13:1 on white.
- **Quiet Surface** (`secondary` / `muted`, oklch(0.968 0.008 75)): wells,
  probability tracks, ingredients block.
- **Muted Ink** (`muted-foreground`, oklch(0.475 0.012 75)): secondary
  text, ≥4.8:1 on white.
- **Soft Border** (`border`, oklch(0.922 0.008 75)): every 1px separation.

### Semantic — FSA nutrient bands (`fsa-*`)
- **Traffic Green** (`fsa-low`, oklch(0.46 0.1 150)) + white icon/word.
- **Traffic Amber** (`fsa-medium`, oklch(0.78 0.11 75)) + **Amber Ink**
  (`on-amber`, oklch(0.22 0.04 75)) label — the only pale-fill chip.
- **Traffic Red** (`fsa-high`, oklch(0.455 0.13 27)) + white.
- All three indicators carry ~25% less chroma than their published FSA
  hues (quieter pass, 25 Sep 2026): bands stay legible, screen never
  shouts. The band word + icon carry meaning; color only confirms it.
- **Not Published** (`fsa-none`, quiet surface + Muted Ink): a `null` band
  never gets a guessed color.

### Semantic — Processing Steps (`nova-*`, neutral cool greys)
- **Step 1** (oklch(0.94 0.004 250)) → **Step 4**
  (oklch(0.32 0.008 250)): light→dark only, L 0.94 / 0.85 / 0.68 / 0.32.
  Used for the NOVA badge, predicted-class probability bar, and history
  chips. Dark steps take white text; light steps take Warm Ink.

### Named Rules
**The Processing Ramp Rule.** The NOVA badge is NEVER green→red — red
reads as "bad," the exact conflation Chewsy exists to fix (binding PRD
rule; PRODUCT.md anti-reference *"Green→red NOVA badge"*). Green/amber/red
is reserved **exclusively** for the four nutrient chips where it matches
the published FSA meaning — and those chips always carry their band word
and icon, never color alone.

**The Honey Budget Rule.** Pressed Honey covers ≤ ~10% of any screen. Its
rarity is what makes an action unmistakable.

## 3. Typography

**Display Font:** Geist (with system-ui fallback)
**Body Font:** Geist (same family — product UI; one tuned grotesque)
**Label/Mono Font:** Geist Mono (with ui-monospace fallback)

**Character:** One neutral grotesque across the whole hierarchy — sizes
and weights do the work, not families. Geist Mono appears exactly where
data must read as data: barcodes, percentages, nutrition values.

### Hierarchy
- **Display** (600, 30px, 1.15, −0.02em): the API `headline` — the
  combined two-axis sentence on the result card. The largest text, always.
- **Title** (600, 18px, 1.375, −0.01em): product name; scan-screen h1 (18–20px).
- **Heading** (600, 14px, 1.4): section titles ("How sure is the model?",
  "Facts per 100g").
- **Body** (400, 14px, 1.5): explainer prose, ingredient list, error copy.
  Prose capped at ~65ch inside the card.
- **Label** (500, 12px, 1.0): axis labels ("How it's made"), nav tagline,
  footer microcopy; uppercase + wide tracking only for the two axis
  captions.
- **Data/Mono** (400, 12–14px): barcodes, confidence %, probability %,
  nutrition values (tabular-nums).

### Named Rules
**The Headline Rule.** The model's combined sentence is the largest text
on the result card — larger than the product name, larger than the NOVA
badge. The card leads with the answer, not the brand.

**The No-Display-Fonts Rule.** One family, fixed rem sizes (no fluid
clamp), 1.125–1.2 scale steps. Display type never enters labels, buttons,
or data tables.

## 4. Elevation

**Flat by default.** This system has no drop shadows at rest — depth is
conveyed by 1px `border` lines, the card ring (`ring-1
ring-foreground/10`), and tonal steps between White Field and Quiet
Surface. The header earns its separation from a border plus a subtle
`backdrop-blur` wash, not a shadow. The result reads like a printed
label: crisp edges, no float.

### Shadow Vocabulary
- *(none — intentionally shadowless)* Hover states change fill or border
  tone, never elevation.

### Named Rules
**The Flat Rule.** Surfaces are flat at rest. If a shadow appears on a
card or button, it is a bug — replace it with a border or a tonal step.
Auditable: "if it looks like it's floating, it's wrong."

## 5. Components

Quietly utilitarian: compact 32px controls, familiar affordances, the
tool disappearing into the task.

### Buttons
- **Shape:** gently curved (12px radius), height 32px (44px minimum on
  coarse pointers — `@media (pointer: coarse)` in globals.css), 14px/500 label,
  10px horizontal padding, leading icon allowed (16px).
- **Primary:** Pressed Honey fill, white label; hover darkens the fill
  (80% tone); active nudges 1px down (press feedback only).
- **Secondary:** Quiet Surface fill, Warm Ink label, hover slightly darkens.
- **Outline:** white fill, 1px Soft Border; hover to Quiet Surface.
- **Focus:** Bench Blue 3px ring at 50% opacity — always visible on
  keyboard focus.
- **Disabled:** 50% opacity, pointer-events off.

### Chips
- **Nutrient chip (signature):** pill (9999px), solid band fill, white or
  Amber-Ink label, 12px/500, always `icon + band word` (CircleCheck /
  TriangleAlert / CircleAlert) so meaning survives color blindness. Never
  used for anything but the four FSA nutrients.
- **NOVA chip:** pill, neutral Processing Steps fill matching the class,
  "NOVA n · Full label" at 14px/500.
- **Confidence chip:** pill — solid Warm Ink + white for "Confident";
  1px outlined for "Fairly sure" and "Borderline" (the weaker tier reads
  as an outline, never as a color warning).
- **History chip:** outlined pill, product name + small ramp-colored NOVA
  square; hover fills Quiet Surface.
- **Positives tag (fibre/protein):** 1px outline, muted 10px text —
  deliberately weaker than nutrient chips so it never impersonates an
  official traffic light.

### Cards / Containers
- **Corner Style:** gently curved (16.8px).
- **Background:** White Field; **Border:** card ring `1px
  ring-foreground/10` (or Soft Border dividers between sections).
- **Shadow Strategy:** none (The Flat Rule).
- **Internal Padding:** 16px base (20–24px at the result card), 20px
  vertical rhythm. Sections are **tonal wells** (`bg-muted/60`, 12px
  radius, 16px padding) with a title-left / muted-caption-right header —
  never nested cards, never hairline dividers (layout pass, 25 Sep 2026).

### Inputs / Fields
- **Style:** height 32px (44px on coarse pointers), 12px radius,
  transparent fill + 1px Soft Border, 14px text (≥16px effective on
  phones via the 17px mobile root — iOS focus-zoom guard); barcode input
  is Geist Mono.
- **Focus:** border shifts to Bench Blue + 3px ring.
- **Error:** red text below field with `role="alert"`; digits-only input
  strips non-numeric keys before state.
- **Disabled:** 50% opacity (submit button until a barcode is typed).

### Navigation
- **Style:** sticky top bar, White Field at 80% + backdrop blur, 1px
  bottom border. Left: 16px SVG barcode-frame mark in Pressed Honey +
  "Chewsy" 14px/600. Right: 12px Muted-Ink tagline "how it's made ·
  what's in it". No nav links — the app is one screen.

### Signature Components
- **Two-axis row:** two tonal wells side by side (stack on mobile) —
  "How it's made" holds the NOVA chip, "What's in it" holds the four
  nutrient chips. The visual grammar of the whole product: two questions,
  never merged.
- **Probability bars:** 10px tracks on Quiet Surface, class name left,
  percentage right (mono), predicted row at 14px/600 with a `PREDICTED`
  tag and its ramp color; others 1.5%-min neutral stubs — ambiguity is
  shown, not summarized.
- **Scan viewfinder:** 4:3 dark (Warm Ink) well, 16.8px radius, white
  reticle from html5-qrcode, bottom strip with state + Stop button;
  permission/insecure/no-device failures swap to an Alert with manual
  entry still live beneath.
- **SHAP chart:** horizontal bars, honey = toward the verdict, Bench
  Blue = away, zero line in ink, muted grid; humanized labels only.

## 6. Do's and Don'ts

### Do:
- **Do** keep Pressed Honey ≤ ~10% of any screen (The Honey Budget Rule).
- **Do** render every processing level on the neutral Processing Steps
  ramp (light→dark) and every nutrient band with `icon + word + color`.
- **Do** keep the API `headline` the largest text on the result card
  (The Headline Rule).
- **Do** render `null` as "—" and say what wasn't published; never guess
  a band or fill a gap with a fabricated number.
- **Do** separate the two axes visually — tonal wells, distinct color
  families, distinct labels.
- **Do** keep surfaces flat with 1px borders; 12px control radius,
  16.8px card radius; 32px control height (The Flat Rule).
- **Do** use Geist Mono for barcodes, percentages, and nutrition values
  (tabular figures).
- **Do** keep all text ≥4.5:1 (ink ≥7:1), honor
  `prefers-reduced-motion`, make every state keyboard-operable, and keep
  touch targets ≥44px on coarse pointers (WCAG 2.5.5).
- **Do** state ambiguity honestly: "Borderline" tier when confidence
  <0.6 or the top-2 gap <15pts.
- **Do** force a "Low information" tier when the API sets
  `data_sparse` — never display a confidence percentage on a record
  with no ingredient list (the number would be a lie by omission);
  the processing axis says "not enough information to classify"
  while nutrition chips keep rendering normally.

### Don't:
- **Don't** ever build the Yuka-style single red/green "health score"
  verdict — we refuse one number by design (two axes, kept separate).
- **Don't** ever color the NOVA badge green→red — "processing level is
  neutral, not a judgment" (binding PRD rule).
- **Don't** borrow calorie-diet-app aesthetics — MyFitnessPal-style
  dashboards, streaks, rings, gamified scoring, guilt-red verdicts.
- **Don't** display any Nutri-Score — OFF's or ours (Hard rules 11 + 12).
- **Don't** commit the slop tells PRODUCT.md names: gradient text,
  health-guru copy, decorative noise — nor wellness-app gradients,
  glassmorphism, side-stripe borders (>1px colored edge), or decorative
  grid/stripe backgrounds.
- **Don't** put drop shadows on cards or buttons — if it floats, it's a bug.
- **Don't** separate card sections with hairline dividers or nest cards
  inside cards — use the tonal Well recipe (`components/Well.tsx`).
- **Don't** round cards past 16.8px, ship buttons taller than 32px for
  decoration, or invent custom scrollbars/modal-heavy flows for standard
  tasks.
- **Don't** use display type in labels/buttons/data, or uppercase
  tracked eyebrows on every section — the two axis captions are the only
  uppercase labels.
- **Don't** recompute any score, band, or headline in the UI — the API is
  the one source of truth.
