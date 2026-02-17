# HyperTensor — UI/UX Build Directive
## "Luxury-Grade Trustless Physics Certification Platform"

**Codename:** OBSIDIAN
**Design Philosophy:** The Bentley of computational physics. Open it and feel like your driver just opened the back door.

---

## THE FEELING

Before any pixel: this is not a dashboard. This is not a dev tool. This is the interface a Head of Verification & Validation at Lockheed Martin opens when $400M of flight certification depends on the answer. It must feel like:

- The weight of a Leica M11 in your hand
- The sound a Bentley Continental door makes when it closes
- The typography in a Patek Philippe catalog
- The silence of a Bloomberg terminal at 4am when you're the only one who knows

**Quality is obvious before comprehension.** A reviewer opens this and thinks "whoever built this gives a shit" within 200 milliseconds, before they read a single word.

---

## I. TYPOGRAPHY — The Voice

Typography is 80% of luxury. Get this wrong and nothing else matters.

### Primary: Serif Display
**Freight Display Pro** or **Canela** for headings, verdicts, and claim titles.
- Weight: Light and Medium only. Never bold. Luxury whispers.
- Size: 42px for page titles, 28px for section heads, 22px for claim titles.
- Letter-spacing: -0.02em (tight, confident, not cramped).
- Color: Off-white (#F5F3EF) on dark, near-black (#1A1A1E) on light.

### Secondary: Monospace for Data
**JetBrains Mono** or **Berkeley Mono** for hashes, values, metrics, code.
- Weight: Regular only.
- Size: 13px. Small. Precise. Like engraved serial numbers.
- Color: Muted — 60% opacity on background. Data is present, not screaming.
- Letter-spacing: +0.04em (let each character breathe like machine engraving).

### Body: Clean Sans
**Söhne** (Klim Type Foundry) or **Untitled Sans** or **Graphik**.
- NOT Inter. NOT Geist. NOT system-ui. Those are free. This is Bentley.
- If licensing is a constraint: **IBM Plex Sans** (free, refined, nobody uses it).
- Weight: Regular (400) for body, Medium (500) for labels. Never bold.
- Size: 15px base. Line-height: 1.65.

### Typography Rules
- Never use ALL CAPS except for tiny labels (11px, +0.12em tracking, 50% opacity).
- Never use bold for emphasis. Use weight contrast between serif and sans instead.
- Numbers in data contexts always use tabular (monospaced) figures.
- Let text breathe. If it feels too close, it is.

---

## II. COLOR — The Palette

Luxury color is restrained. One dominant mood. One accent that means something.

### Dark Mode (Primary — this is the default)

```
--obsidian:        #0D0D10     /* Not black. Deep blue-black, like polished obsidian */
--obsidian-raised: #16161B     /* Cards, panels — barely lifted */
--obsidian-hover:  #1E1E25     /* Hover states — subtle warmth */
--surface:         #232329     /* Active selections, input fields */

--text-primary:    #F5F3EF     /* Warm off-white. Never pure white. */
--text-secondary:  #9994A1     /* Muted lavender-grey */
--text-tertiary:   #5C5866     /* Whisper-level labels */

--accent-gold:     #C9A96E     /* The Bentley accent. Used ONCE per screen. */
--accent-gold-dim: rgba(201, 169, 110, 0.15)  /* Gold glow, very subtle */

--verdict-pass:    #3D8B5E     /* Deep forest — not neon green */
--verdict-fail:    #A8423F     /* Muted crimson — not alarm red */
--verdict-warn:    #B8862D     /* Aged gold — not yellow */
--verdict-neutral: #5C5866     /* Same as tertiary text */

--border:          rgba(255, 255, 255, 0.06)  /* Barely there. Like a seam in leather. */
--border-active:   rgba(201, 169, 110, 0.25)  /* Gold thread on focus */
```

### Light Mode (Secondary — for print and well-lit environments)

```
--obsidian:        #FAFAF8     /* Warm paper */
--obsidian-raised: #FFFFFF     /* Clean white cards */
--surface:         #F0EFEC     /* Subtle warmth */

--text-primary:    #1A1A1E
--text-secondary:  #6B6777
--text-tertiary:   #9994A1

--accent-gold:     #8B7340     /* Deeper gold for contrast on light */
```

### Color Rules
- **Gold is sacred.** It marks: verified attestations, the single most important element on screen, and nothing else. If gold is everywhere, it means nothing.
- Status colors are muted, never saturated. This isn't a video game. Failures are stated, not screamed.
- No gradients. No glows. No neon. Gradients are for crypto bros. This is for engineers who sign off on flight hardware.
- Borders are almost invisible. Structure comes from spacing and elevation, not lines.

---

## III. SPACING & LAYOUT — The Silence

Luxury is knowing what to leave out. The whitespace IS the design.

### Spacing Scale (8px base)
```
--space-xs:   4px      /* Inside tight elements */
--space-sm:   8px      /* Between related items */
--space-md:   16px     /* Default padding */
--space-lg:   24px     /* Between sections */
--space-xl:   40px     /* Major section breaks */
--space-2xl:  64px     /* Page-level breathing room */
--space-3xl:  96px     /* Hero-level silence */
```

### Layout Principles
- **Page margins: 48px minimum.** Let the edges breathe.
- **Card padding: 32px.** Not 16. Not 24. 32. The content floats inside.
- **Between cards: 16px.** Tight enough to group, loose enough to distinguish.
- **Between sections: 64px.** A full breath between ideas.
- **Max content width: 1400px.** Even on a 4K monitor, the content stays centered and composed. No edge-to-edge sprawl.
- **The 4-column cockpit uses a 280px left rail, flexible center, 320px right rail.** The center is king.

### Density Toggle
Two modes. Both luxurious, different postures.

- **Review Mode (default):** Generous spacing. One claim fills the viewport. Reading a fine book.
- **Audit Mode:** Tighter spacing, more rows visible, comparison panels open. Bloomberg density. Still elegant — smaller type, tighter grid, but every rule above still applies at smaller scale.

---

## IV. ELEVATION & DEPTH — The Weight

No drop shadows. No box-shadows with blur. That's 2019.

### How Things Float
```css
/* Level 0: The void. Page background. */
background: var(--obsidian);

/* Level 1: Cards, panels. Barely lifted. */
background: var(--obsidian-raised);
border: 1px solid var(--border);

/* Level 2: Active selection, expanded claim. */
background: var(--surface);
border: 1px solid var(--border-active);

/* Level 3: Modal, command palette, dropdown. */
background: var(--obsidian-raised);
border: 1px solid var(--border);
backdrop-filter: blur(24px) saturate(1.2);
```

- Elevation comes from **background shade + border subtlety**, not shadow.
- The only shadow in the entire app is on **modals**: `0 24px 80px rgba(0,0,0,0.5)`. One shadow. Massive. Cinematic.
- Glass (`backdrop-filter`) is used ONLY on overlays (command palette, modals). Never on cards. Never on panels.

---

## V. MOTION — The Weight of Movement

Everything moves. Nothing bounces. Nothing overshoots. Things arrive with the inevitability of a closing vault door.

### Easing
```css
--ease-out:     cubic-bezier(0.16, 1, 0.3, 1);      /* Primary. Deceleration. Arrival. */
--ease-in-out:  cubic-bezier(0.65, 0, 0.35, 1);      /* Transitions between states. */
--ease-subtle:  cubic-bezier(0.25, 0.1, 0.25, 1);    /* Hover states. Almost linear. Barely there. */
```

### Durations
```
Hover states:          120ms   /* Instant acknowledgment */
Panel switches:        250ms   /* Deliberate but not slow */
Page transitions:      400ms   /* The door closing */
Proof verification:    800ms   /* Ceremony. See below. */
Loading states:        ∞       /* Skeleton shimmer, never spinners */
```

### The Verification Ceremony
When a proof package is opened and verified:
1. **0ms:** Page loads with skeleton structure visible.
2. **200ms:** Identity strip fades in. Monospace hashes appear character by character, left to right, like a typewriter. 50ms per character group.
3. **400ms:** Proof tree fades in from left. Claims appear with staggered delay (30ms each).
4. **600ms:** The integrity badge begins its check. A thin gold line traces the perimeter of the badge, like a wax seal forming.
5. **800ms:** The line completes. The badge fills: **deep forest green** for verified, **muted crimson** for failed.
6. **1000ms:** If verified, the accent-gold briefly pulses once on the badge border. Once. Then settles. The Bentley headlights acknowledging you.

This is the moment. This is what they remember. This 1-second ceremony IS the brand.

### Anti-Patterns
- No bounce. No elastic. No spring physics on UI elements. Springs are for Figma demos, not flight certification.
- No parallax. No scroll-triggered animations on content. The content is too important to make it dance.
- No loading spinners. Skeleton screens with a slow shimmer (3s cycle, 5% opacity range).
- No confetti. No celebrations. A quiet gold pulse is the maximum expression of success.

---

## VI. COMPONENTS — The Craft

### The Verdict Badge
The most important 48x48 pixels in the entire application.

```
PASS:  Circle. 2px border: --verdict-pass. Interior: transparent.
       Center: thin check mark (1.5px stroke). Not filled. Not fat.
       On hover: interior fills with --verdict-pass at 10% opacity.

FAIL:  Circle. 2px border: --verdict-fail. Interior: transparent.
       Center: thin × (1.5px stroke).

WARN:  Circle. 2px border: --verdict-warn. Interior: transparent.
       Center: thin ! (1.5px stroke).
```

- These badges are NEVER filled solid. The restraint is the luxury.
- They are NEVER accompanied by text like "PASSED!" — the icon is sufficient.
- Size: 20px inline with claims, 32px on summary cards, 48px on the run overview.

### The Claim Card
The atomic unit. Must be perfect.

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│  ○ Conservation of Total Energy           CLM-042   │
│                                                     │
│  Total system energy drift must remain below        │
│  0.1% over the simulation window.                   │
│                                                     │
│  ┌─────────────────────────────────────────────┐    │
│  │  GATE                                       │    │
│  │                                             │    │
│  │  |E(t_f) - E(t_0)| / E(t_0)  <  0.001     │    │
│  │                                             │    │
│  │  Observed: 0.0004    Threshold: 0.001       │    │
│  │  Margin: 60%                                │    │
│  └─────────────────────────────────────────────┘    │
│                                                     │
│  Evidence  2 artifacts    Dependencies  CLM-039     │
│                                                     │
└─────────────────────────────────────────────────────┘
```

- Background: `--obsidian-raised`. Border: `--border`.
- Claim title: Serif. 18px. Medium weight.
- Claim ID: Monospace. 11px. `--text-tertiary`. Top right.
- Description: Sans. 14px. `--text-secondary`. 2 lines max, truncate with "..."
- Gate block: `--surface` background. Monospace for the formula. Sans for labels.
- Observed/Threshold: Monospace. Color-coded by margin (green if >50%, gold if 10-50%, red if <10%).
- Bottom row: Sans. 12px. `--text-tertiary`. Links to evidence and dependencies.
- On hover: border transitions to `--border-active` (120ms).
- On select: left border becomes 2px `--accent-gold`. That's it. No other decoration.

### The Identity Strip (Top Bar)
Fixed. Always visible. The chassis plate.

```
 Project Name / Solver v2.1.0 / Run #8492          sha256:7b3f1a...  ●  Seed: 42  ●  ✓ Verified
```

- Height: 48px exactly. Feels like a thin, machined aluminum bar.
- Background: `--obsidian` (same as page — it floats by position, not color).
- Bottom border: 1px `--border`.
- Left: Project path in sans. Medium weight. `--text-primary`.
- Right: Hash (monospace, truncated, click to copy), seed, integrity badge.
- The integrity badge here is 16px. Tiny. Confident. Like a hallmark stamp on silver.

### The Proof Tree (Left Rail)
Navigation. Quiet until needed.

- Width: 280px. Collapsible to 48px (icon-only).
- Each node: 36px height. Verdict badge (16px) + claim title (sans, 13px).
- Indent: 20px per level. Thin vertical lines connecting children (1px, `--border`).
- Search at top: no border input. Just a monospace placeholder: `Search claims...` at 40% opacity.
- Selected node: left edge gets 2px `--accent-gold` bar. Background: `--surface`.
- Collapsed categories show count badge: monospace, 11px, `--text-tertiary`.

### Command Palette
⌘K to open. The power move.

- Full-width overlay, vertically centered, max-width 640px.
- `backdrop-filter: blur(24px)`.
- Single input field. Serif placeholder: `Navigate to...`
- Results: instant, fuzzy. Show type chips (Claim, Run, Artifact, Gate) in monospace, 10px, uppercase.
- No mouse required. Arrow keys + Enter.

---

## VII. THE SIX SCREENS

### Screen 1: Proof Package Overview
**Purpose:** First impression. 5-second comprehension.

- Center: Run verdict (48px badge) + project name (serif, 36px) + one-line summary.
- Below: 3-5 invariant tiles. Each tile: metric name (sans, 12px label), value (monospace, 28px), unit (monospace, 12px, tertiary), tiny sparkline if time-series.
- Below: Compression summary (observed rank range, memory ratio, wall time).
- Below: Integrity summary (hashes, verification status, attestation chain).
- All on a single scrollable page. No tabs. The overview IS the experience.

### Screen 2: Run Detail (The World-Class Screen)
**Purpose:** Claim-by-claim review. The cockpit.

- 4-column layout per the wireframe spec.
- Left: Proof tree.
- Center-left: Claim card (selected claim, full detail).
- Center-right: Evidence viewer (typed: time-series, field, spectrum, log).
- Right: Gates + Reproduce + Integrity panel.
- Keyboard navigation: ↑↓ moves through claims, → opens evidence, ← returns to tree.

### Screen 3: Step Timeline
**Purpose:** Temporal navigation across the simulation.

- Horizontal timeline (top) with step markers. Anomalies flagged with amber dots.
- Below: Metric strip charts (small multiples). Each row = one metric across all steps.
- Click any step: claim cards below update to show that step's values.
- State hash displayed per step in monospace. Hash chain visualized as connected dots.

### Screen 4: Compare Runs
**Purpose:** Baseline vs candidate. What changed and why.

- 3-column: Baseline | Delta | Candidate.
- Parameter diff at top (semantic, not text).
- Metric comparison: side-by-side values with delta column.
- Plot overlays: both runs on same axes, with residual plot below.
- Environment diff: container digest, commit, compile flags.

### Screen 5: Reproduce
**Purpose:** One-click reproduction. The trust contract.

- Single command block at top. Monospace. Click to copy.
- Below: Environment lock (container digest, commit hash, seeds, dependencies).
- Below: Expected outputs (artifact hashes, metric tolerances).
- Below: "Verify Only" mode (re-hash artifacts locally, no recompute).
- This page should feel like a legal document. Clean. Definitive. Nothing decorative.

### Screen 6: Integrity Explorer
**Purpose:** Cryptographic provenance for the paranoid.

- Merkle tree visualization. Each node = artifact hash. Click to inspect.
- Signature chain: who signed, when, with what key, verification status.
- Impact mapping: if any artifact is tampered, show which claims are invalidated.
- The tree visualization: thin lines, small nodes (8px circles), monospace labels. NOT a flashy graph visualization. Quiet. Precise. Like a circuit diagram.

---

## VIII. DOMAIN LENS SYSTEM

One UI. 140 domains. No fragmentation.

### The Lens Switcher
- Top-left, next to project name in identity strip.
- Dropdown: search-first. Type to filter. Grouped by Family.
- Shows current lens name + a subtle icon.
- Switching lenses: 250ms crossfade. Invariant tiles, plot presets, and gate defaults update. Structure doesn't change.

### What a Domain Pack Provides
```json
{
  "domain": "vlasov_maxwell",
  "family": "Kinetic",
  "display_name": "Vlasov-Maxwell (6D)",
  "headline_invariants": ["total_mass", "total_energy", "l2_norm", "gauss_law_residual"],
  "gate_pack": "vlasov_gates_v1",
  "viz_presets": {
    "phase_space_slice": { "type": "field_2d", "axes": ["x", "vx"], "colormap": "inferno" },
    "field_energy": { "type": "time_series", "metrics": ["E_field_energy", "B_field_energy"] },
    "distribution_function": { "type": "distribution", "moments": [0, 1, 2] }
  },
  "symbol_table": {
    "f": { "type": "scalar_field", "dims": 6, "units": "phase_space_density" },
    "E": { "type": "vector_field", "dims": 3, "units": "V/m" },
    "B": { "type": "vector_field", "dims": 3, "units": "T" }
  },
  "assumptions": ["collisionless", "non-relativistic", "periodic_boundaries"]
}
```

### Lens Rules
- Lens changes content, never structure.
- If a metric is missing for a domain, the tile shows "—" with `--text-tertiary`. Never hide the tile. The absence is information.
- Cross-domain comparison works because the gate manifest is standardized. Domains add metrics; the comparison engine diffs whatever overlaps.

---

## IX. INTERACTIONS THAT DEFINE THE BRAND

### Hover-to-Trace
- Hover any metric value → its source in the evidence pane highlights with a 120ms gold underline.
- Hover a claim → its dependencies in the proof tree get a subtle pulse (one cycle, --accent-gold at 8% opacity).
- Hover a hash → full hash appears in a tooltip. Monospace. No decoration. Click to copy.

### Progressive Disclosure
- Default: verdict + margins. One click: gate details + evidence. Two clicks: raw artifacts.
- Never force three clicks to reach truth. Two is the maximum depth.

### Keyboard First
- `↑↓` navigate claims. `Enter` opens evidence. `Esc` returns. `⌘K` command palette.
- `D` toggles density mode. `L` toggles light/dark. `C` opens compare.
- These shortcuts shown in a quiet `--text-tertiary` hint at bottom of the proof tree on first visit. Then never again.

### Copy Behavior
- Click any hash: copies full hash. Brief gold flash on the element (80ms).
- Click any command: copies to clipboard. Monospace text briefly inverts (gold text on obsidian).
- No "Copied!" toast. No tooltip. The flash IS the confirmation. Quiet confidence.

---

## X. WHAT THIS IS NOT

- Not a dashboard with charts. No pie charts. No bar charts. No "analytics."
- Not a dark mode template with purple gradients.
- Not a "futuristic" sci-fi UI with glowing edges and particle effects.
- Not a Material Design app with floating action buttons.
- Not a glass-morphism showcase with stacked blur layers.
- Not a tool that looks like it was built by developers for developers.

This is a tool that looks like it was built by people who understand that when the answer matters — when flight certification, reactor safety, or fusion research depends on it — **the interface must be as trustworthy as the mathematics.**

---

## XI. THE ONE-LINE TEST

If a reviewer opens this and doesn't instinctively sit up straighter, it's not done.

---

*Directive for HyperTensor OBSIDIAN UI*
*© 2026 Brad McAllister / Tigantic Holdings. All rights reserved.*
