# SANKET v1 — UI/UX Design Brief

## 1. Direction

Calm, high-clarity, "two people at one counter" feel. The screen is shared by two people sitting/standing opposite each other, so **text must be readable from 1.5 m away** and nothing important may depend on colour alone. Warm neutral background, one accent colour (deep teal `#0F6E6E`), confidence colours green / amber / red **always paired with an icon + word**.

Typography: Noto Sans + Noto Sans Devanagari (bundled, OFL). Caption size default 28 px, adjustable 20–48 px.

## 2. Information architecture

```
Conversation (home)   ← opens here, always
Dictionary            ← browse / search 263 signs, avatar playback
Settings              ← camera, mic, voice, avatar speed, threshold, text size, theme, UI language
   └─ Help improve SANKET (Contribute, off by default)
About                 ← credits (INCLUDE signers, CC BY 4.0), licences, KP logo, version, "Preview" note
```
Left icon rail (4 items) on wide windows; bottom tab bar under 900 px width.

## 3. Conversation screen layout (≥ 1280 × 720)

```
┌──────────────────────────────────────────────────────────────────────────┐
│ SANKET  ● Offline   Preview · 263 words                        [KP logo] │
├───────────────────────────────┬──────────────────────────────────────────┤
│  SIGNER (camera)              │  SPEAKER (avatar)                        │
│  ┌─────────────────────────┐  │  ┌────────────────────────────────────┐  │
│  │ live video + landmarks  │  │  │ 2D avatar signing                  │  │
│  │                         │  │  │                                    │  │
│  └─────────────────────────┘  │  └────────────────────────────────────┘  │
│  chips: [AAP ✓92%] [NAAM ✓]   │  gloss: AAP NAAM KYA  ▶ current word     │
│  "Aapka naam kya hai?"  🔊    │  [ 🎤  Hold to speak (Space) ]           │
├───────────────────────────────┴──────────────────────────────────────────┤
│ Transcript  (newest at bottom, big text, both languages, session only)   │
└──────────────────────────────────────────────────────────────────────────┘
```
"Flip" button swaps left/right so the avatar faces whoever sits on that side.

## 4. Key interactions

- Sign recognised → chip pops in (150 ms); low confidence → amber chip with "?" and top-2 alternative on hover/tab focus; tapping an alternative replaces it.
- Sentence closes → text slides up into transcript, TTS plays, 🔊 button replays.
- Hold Space / button → mic level meter; release → "Samajh raha hoon…" spinner → avatar plays; current caption word underlined.
- Esc = clear current sentence. Enter = close sentence now. Ctrl+F = flip sides.

## 5. States (every screen)

| State | Conversation screen shows |
|---|---|
| Loading | Logo + "Models load ho rahe hain…" progress (≤ 3 s target) |
| Empty | Ghost text: "Camera ke saamne sign karein" / "Bolne ke liye Space dabaayein" |
| Working | Live overlay, chips, mic meter |
| Low confidence | Amber chip + "Samajh nahi aaya — dobara karein" |
| Error (no camera / mic) | Inline banner with Retry; other direction stays usable |
| Disabled | Speak button greyed with tooltip when no mic |
| Success | Green tick on chip, sentence in transcript |

## 6. Accessibility

WCAG 2.1 AA contrast (≥ 4.5:1 text), visible focus ring, full keyboard path, accessible names on all controls, no flashing > 3 Hz, captions never auto-disappear while being read (min 4 s), reduced-motion setting disables chip animations. High-contrast theme (black/white/yellow).

## 7. Responsive

≥ 1280 px: side-by-side. 900–1279 px: stacked panels, smaller video. < 900 px: tabs "Sign" / "Speak" (prepares Android layout for v2).

## 8. Brand

KP orbit logo (`assets/logo-mark.svg`) 24 px in the title bar corner + About. App icon: a stylised open palm with a speech arc, teal on white (to be designed; placeholder allowed in dev, real icon required before Store).
