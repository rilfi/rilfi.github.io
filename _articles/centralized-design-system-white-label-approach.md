---
title: "The Complete Guide to Building a Multi-Brand + Dark Mode Design System in Next.js, Tailwind & TypeScript"
categories: ["Design Systems", "Frontend Architecture"]
layout: default
---


# The Complete Guide to Building a Multi-Brand + Dark Mode Design System in Next.js, Tailwind & TypeScript

(With Build-Time Brand Switching)**
This is the full guide for building a **scalable, maintainable, multi-brand design system** with:

* ✔ Next.js
* ✔ Tailwind CSS
* ✔ TypeScript design tokens
* ✔ Build-time brand switching
* ✔ Light + Dark mode
* ✔ Semantic token architecture
* ✔ Numeric color scales (50–900)
* ✔ Brand/neutral/feedback/accent categories
* ✔ Full project folder structure

---

## **PRE-STEP 1 — Collect Design Tokens**

Ask designers for:

### Colors:

* Brand palette
* Neutral palette
* Accent palette (optional)
* Feedback palette (success / warning / error / info)
* Dark mode palette if available

### Typography:

* Header font
* Body font
* Line heights
* Weight scale
* Base font size

### Spacing:

* Base grid (usually 4px or 8px)
* Component spacing used in Figma

### UI Properties:

* Border radius scale
* Shadows
* Icon rules
* Motion (optional)

---

## **PRE-STEP 2 — Decide Token Categories**

Use these recommended buckets:

| Category                | Purpose                        | Example Tokens          |
| ----------------------- | ------------------------------ | ----------------------- |
| **brand**               | Identity colors                | brand.500, brand.700    |
| **neutral**             | Backgrounds, text, surfaces    | neutral.50, neutral.900 |
| **feedback**            | Success/error/warning/info     | success.700             |
| **accent** (optional)   | Secondary brand colors         | accent.500              |
| **utility**             | Overlay, focus rings, skeleton | overlay, focus          |
| **gradient** (optional) | Brand gradient tokens          | gradient.primary        |
| **typography**          | Body, header, weights          | font-body               |
| **spacing**             | Layout spacing                 | spacing-200             |
| **radius**              | Border radii                   | radius-md               |

---

## **PRE-STEP 3 — Agree on Semantic Token Naming**

Semantic tokens represent *meaning*, not raw values.

Recommended naming:

### Text

```
text-body
text-title
text-subtle
text-disabled
text-critical
```

### Surfaces

```
surface-default
surface-alt
surface-accent
surface-critical
```

### Borders

```
border-default
border-strong
border-critical
```

### Interactions

```
interactive-primary-default
interactive-primary-hovered
interactive-primary-pressed
```

### Sizes

```
spacing-xs
spacing-m
radius-lg
```

### Dark mode (prefix “dark.”)

```
dark.text-body
dark.surface-default
dark.interactive-primary-default
```

---

---

# 📁 **PART 1 — FULL PROJECT FILE STRUCTURE (HIGHLY RECOMMENDED)**

This structure is optimized for:

* clear separation
* scalable brand addition
* reusable components
* clean Tailwind integration
* maintainability

---

# **📦 Root folder structure**

```
project/
│
├── config/
│   ├── brands/
│   │   ├── brand-one/
│   │   │   ├── core.ts                # Raw values
│   │   │   ├── colors.light.ts        # Semantic tokens (light)
│   │   │   ├── colors.dark.ts         # Semantic tokens (dark)
│   │   │   ├── typography.map.ts      # Typography tokens
│   │   │   ├── spacing.map.ts         # Spacing tokens
│   │   │   └── index.ts               # Re-export
│   │   ├── brand-two/
│   │   │   ├── core.ts
│   │   │   ├── colors.light.ts
│   │   │   ├── colors.dark.ts
│   │   │   ├── typography.map.ts
│   │   │   ├── spacing.map.ts
│   │   │   └── index.ts
│   │   └── brand-three/
│   │       └── ...
│   │
│   ├── tailwind-brand-loader.ts        # Decides brand from .env
│   ├── spacing/
│   │   └── base-spacing.ts             # Global grid system
│   ├── constants.ts                    # PRODUCT, SECONDARY_PRODUCT
│   └── tokens.types.ts                 # Optional TS types for tokens
│
├── styles/
│   ├── globals.css                     # Tailwind base, resets
│   ├── dark-mode.css                   # Dark mode class strategy
│   └── tailwind.css                    # Tailwind imports
│
├── tailwind.config.ts                  # Final Tailwind integration
├── postcss.config.js
├── next.config.js
├── tsconfig.json
│
├── .env                                # PRODUCT=brandOne
│
├── src/
│   ├── app/ or pages/
│   │   ├── layout.tsx
│   │   ├── page.tsx
│   │   └── ...
│   │
│   ├── components/
│   │   ├── primitives/                 # Button, Input, Card, etc.
│   │   ├── feedback/                   # Alert, Toast, Banner
│   │   ├── navigation/                 # Header, NavBar, Sidebar
│   │   ├── forms/                      # Select, Checkbox, Radio
│   │   └── index.ts
│   │
│   ├── hooks/
│   │   └── useDarkMode.ts              # Optional hook
│   │
│   ├── utils/
│   │   └── classnames.ts
│   │
│   ├── lib/
│       └── brand-context.ts            # Optional runtime brand info
│
└── tests/
    ├── visual/                         # Visual regression tests
    ├── tokens/                         # Validate token structure
    └── components/
```

---


# 🎨 PART 2 — UNDERSTANDING NUMERIC COLOR SCALES (50 → 900)

Most modern design systems use numeric scales:

```
50, 100, 300, 500, 700, 900
```

### Why this scale?

### 1️⃣ Represents a controlled **light → dark spectrum**

* 50 = lightest
* 900 = darkest

### 2️⃣ Works for all palettes

* neutrals
* brand colors
* feedback colors

### 3️⃣ Follows Tailwind, Material Design, Radix Colors, IBM Carbon, Adidas, Shopify

This makes your system feel industry-standard.

### 4️⃣ Supports dark mode cleanly

`text-body` switches from `neutral.900` → `neutral.50`

### 5️⃣ You can skip numbers

Design doesn’t always give 9 shades. You can map:

```
light → 100  
mid → 300  
dark → 900
```

The numbering stays meaningful.

---

# 🧱 PART 3 — THREE LAYER TOKEN ARCHITECTURE (WITH EXAMPLES)

This is the core of the system.

```
Core Tokens (TS)
    ↓
Semantic Tokens (TS)
    ↓
Tailwind (build-time)
```

---

# LAYER 1: CORE TOKENS (RAW VALUES)

### ✔ What this layer is

Brand-specific raw values directly from Figma.

### ✔ Why it exists

* Keeps raw colors isolated per brand
* Easy for designers to update
* Never imported directly by components
* Supports light & dark color values separately

### ✔ What files you create

```
config/brands/brand-one/core.ts
config/brands/brand-two/core.ts
```

---

### 🔧 Example Core Token File (Brand One)

```ts
export const brandOneCore = {
  colors: {
    brand: {
      100: "#EFEAFF",
      300: "#D6C7FF",
      500: "#A88CFF",
      700: "#6B3EFF",
      900: "#2B0D9E",
    },
    neutral: {
      50: "#FFFFFF",
      100: "#F9FAFB",
      300: "#E5E7EB",
      600: "#6B7280",
      900: "#111827",
    },
    feedback: {
      success100: "#ECFDF5",
      success700: "#047857",
      error100: "#FEF2F2",
      error700: "#B91C1C",
    },
  },
  typography: {
    body: '"Inter", sans-serif',
    header: '"Poppins", sans-serif',
    baseFontSize: "16px",
  },
  spacing: {
    100: "8px",
    150: "12px",
    200: "16px",
    300: "24px",
  },
  radius: {
    sm: "4px",
    md: "8px",
    lg: "12px",
  },
};
```

---

# LAYER 2: SEMANTIC TOKENS (LIGHT + DARK)

### ✔ What this layer is

Maps brand values → meaningful names.

### ✔ Why it exists

* Components never break when brands change
* Dark mode values live here
* Designers and developers speak the same language

### ✔ Files you create

```
config/brands/brand-one/colors.map.ts
config/brands/brand-one/colors-dark.map.ts
```

---

### 🌞 Light Mode Mapping (Example)

```ts
import { brandOneCore } from "./core";

export const brandOneLight = {
  "text-body": brandOneCore.colors.neutral[900],
  "text-title": brandOneCore.colors.brand[700],
  "surface-default": brandOneCore.colors.neutral[50],
  "border-default": brandOneCore.colors.neutral[300],
  "interactive-primary-default": brandOneCore.colors.brand[700],
};
```

---

### 🌙 Dark Mode Mapping (Example)

```ts
export const brandOneDark = {
  "text-body": brandOneCore.colors.neutral[50],
  "text-title": brandOneCore.colors.brand[300],
  "surface-default": brandOneCore.colors.neutral[900],
  "border-default": brandOneCore.colors.neutral[600],
  "interactive-primary-default": brandOneCore.colors.brand[300],
};
```

---

# LAYER 3: BUILD-TIME BRAND LOADER

### ✔ Purpose

Pick the correct brand files based on `.env`.

### ✔ Why build-time?

* CSS bundle is small
* No runtime overhead
* Fully isolated builds
* CI can build each brand separately

---

### 🔧 Example: `tailwind-brand-loader.ts`

```ts
import { brandOneLight, brandOneDark } from "./config/brands/brand-one/colors.map";
import { brandTwoLight, brandTwoDark } from "./config/brands/brand-two/colors.map";

const PRODUCT = process.env.PRODUCT;
const SECONDARY_PRODUCT = process.env.SECONDARY_PRODUCT;

export const brandKey =
  SECONDARY_PRODUCT === "brandTwo"
    ? "brandTwo"
    : (PRODUCT as "brandOne" | "brandTwo") || "brandOne";

export const selectedBrandTokens = {
  light: {
    brandOne: brandOneLight,
    brandTwo: brandTwoLight,
  }[brandKey],
  dark: {
    brandOne: brandOneDark,
    brandTwo: brandTwoDark,
  }[brandKey],
};
```

---

# LAYER 4: TAILwind CONFIG

### ✔ What it does

* Injects semantic tokens
* Supports dark mode
* Tailwind compiles only selected brand

---

### 🔧 Example: `tailwind.config.ts`

```ts
import type { Config } from "tailwindcss";
import { selectedBrandTokens } from "./tailwind-brand-loader";

const config: Config = {
  darkMode: "class",
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ...selectedBrandTokens.light,
        dark: { ...selectedBrandTokens.dark },
      },
    },
  },
};
export default config;
```

---

# 🎛 PART 4 — COMPONENT IMPLEMENTATION

### Example: Button (light + dark)

```tsx
export const Button = () => (
  <button
    className="
      bg-interactive-primary-default
      hover:bg-interactive-primary-hovered
      text-text-body
      dark:bg-dark-interactive-primary-default
      dark:text-dark-text-body
      px-spacing-200 py-spacing-150 rounded-md
    "
  >
    Continue
  </button>
);
```

---

### Example: Card

```tsx
export const Card = ({ title, children }) => (
  <div
    className="
      bg-surface-default 
      dark:bg-dark-surface-default 
      border 
      border-border-default 
      dark:border-dark-border-default 
      p-spacing-300 rounded-lg
    "
  >
    <h2 className="text-text-title dark:text-dark-text-title">{title}</h2>
    <p className="text-text-body dark:text-dark-text-body">{children}</p>
  </div>
);
```

---

# 🛠 PART 5 — STEP-BY-STEP IMPLEMENTATION WORKFLOW

---

# 🧩 WORKFLOW A — Starting from a Figma design

### ✅ Step 1 — Extract raw design tokens

Export:

* brand palette
* neutrals
* feedback colors
* typography styles
* spacing grid
* radii

### ✅ Step 2 — Build `core.ts`

Paste raw values.

### ✅ Step 3 — Build semantic maps

Map each raw color to a meaning.

### Example:

```
surface-default → neutral.50  
text-body → neutral.900
```

### ✅ Step 4 — Add dark mode semantics

Ask design which pairs invert.

### Example:

```
surface-default → neutral.900  
text-body → neutral.50  
```

### ✅ Step 5 — Implement brand loader

PRODUCT decides theme.

### ✅ Step 6 — Add Tailwind integration

Add semantic tokens in `tailwind.config.ts`.

### ✅ Step 7 — Component migration

Start with:

* Button
* Card
* Input
* Alerts

### ⚡ Critical rule:

**Components NEVER use colors directly. Only semantic classes.**

---

# 🔧 WORKFLOW B — Migrating an Existing Website

### Step 1 — Extract all hex colors

Use grep:

```
grep -R "#[0-9a-fA-F]" -n src
```

### Step 2 — Group colors

Organize them into:

* brand
* neutral
* feedback
* accent

### Step 3 — Create numeric scale (50 → 900)

Sort from light to dark.

### Step 4 — Build `core.ts`

Add raw palette.

### Step 5 — Create semantic tokens

Map meaning → raw color.

### Step 6 — Codemod migration

Replace raw hex with semantic classes.

### Step 7 — Add dark mode

Define dark semantic maps.

### Step 8 — Full regression testing

Use Playwright + visual snapshots.

---

# 🎉 PART 6 — Adding a New Brand in 5 Minutes

1. Duplicate `config/brands/brand-one/` → `brand-three/`
2. Replace raw colors in `core.ts`
3. Update semantic map files
4. Add brandThree to brand loader
5. Build:

```
PRODUCT=brandThree npm run build
```

Done.

---

# 🧭 PART 7 — Frequently Used Color Categories (With Explanation)

### **neutral**

Backgrounds, body text, borders, surfaces.
Should NOT represent brand identity.

### **brand**

Brand personality colors for titles, highlights, CTAs.

### **feedback**

System messages (success, error, warning, info).
Consistent across brands but shades may vary.

### **accent**

Optional additional tones (secondary brand color).

### **utility**

Technical colors (overlay, scrim, focus rings, shadows).

### **gradient**

Optional for brands using gradient identity.

---

# 🎯 Final Benefits

* One codebase, unlimited brands
* Dark mode baked into tokens
* Super maintainable design system
* No runtime overhead
* Smaller CSS bundles
* Perfect Tailwind + TypeScript integration
* Designers update tokens → entire UI updates
* Seamless onboarding for new developers

