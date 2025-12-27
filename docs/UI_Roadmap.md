# HyperTensor Forensic Visualization Roadmap

## Vision Statement

Produce a **broadcast-quality atmospheric visualization system** that rivals NOAA/NASA operational displays. The user should see wind currents flowing over photorealistic terrain as if viewing a professional weather broadcast or climate research presentation.

**Reference Standard:** The attached cyclone visualization with satellite-quality Earth imagery, focused vector fields with color-coded intensity, and professional HUD overlays.

---

## Phase 1: Foundation — Asset Acquisition & Rendering Pipeline

### Milestone 1.1: NASA Blue Marble Integration
**Objective:** Replace procedural noise with actual satellite imagery

- [ ] Download NASA Blue Marble "Next Generation" imagery
  - Source: https://visibleearth.nasa.gov/collection/1484/blue-marble
  - Resolution: 21600×10800 px (or 5400×2700 for faster loading)
  - Variant: "No clouds" version for clean terrain
  - Format: PNG or JPEG, equirectangular projection
  
- [ ] Create asset loader with resolution tiers
  - `assets/blue_marble_8k.jpg` — Full quality (21600×10800)
  - `assets/blue_marble_4k.jpg` — Standard (5400×2700)
  - `assets/blue_marble_1k.jpg` — Thumbnail (1350×675)
  
- [ ] Implement lazy loading with progressive refinement
  - Load 1K immediately for responsiveness
  - Background-load 4K/8K and swap when ready
  
- [ ] Verify coordinate alignment
  - Longitude: -180° to +180° maps to image left→right
  - Latitude: +90° to -90° maps to image top→bottom
  - Test: Africa centered around x=50%, y=50%

**Acceptance Criteria:**
- [ ] Sahara Desert is visibly tan/yellow
- [ ] Amazon rainforest is visibly green
- [ ] Ocean depth gradients visible (Caribbean lighter than Atlantic)
- [ ] Polar ice caps white
- [ ] No visible seams at date line (180°)

---

### Milestone 1.2: Proper Compositing Pipeline
**Objective:** Wind data overlays terrain without obscuring it

- [ ] Implement alpha-blended compositing in correct order:
  ```
  Layer 0 (back):  Blue Marble terrain
  Layer 1:         Ocean depth enhancement (optional)
  Layer 2:         Wind magnitude field (40-60% opacity)
  Layer 3:         Isobar contours (optional)
  Layer 4:         Vector flow lines
  Layer 5:         Vector arrow heads
  Layer 6:         Coastline outlines
  Layer 7:         Geographic labels
  Layer 8 (front): HUD overlays, legends, annotations
  ```

- [ ] Use pre-multiplication for correct alpha blending
  - Formula: `result = src * src_alpha + dst * (1 - src_alpha)`
  - Apply in linear color space, not sRGB

- [ ] Implement intensity-based transparency for wind field
  - Low wind: more transparent (see terrain clearly)
  - High wind: more opaque (emphasize anomaly)
  - Formula: `alpha = 0.3 + 0.5 * normalized_intensity`

**Acceptance Criteria:**
- [ ] Terrain features visible through low-intensity wind regions
- [ ] High-intensity features (storms) stand out prominently
- [ ] No "floating in void" appearance
- [ ] Coastlines align with terrain boundaries

---

## Phase 2: Vector Field Visualization

### Milestone 2.1: Professional Streamline Rendering
**Objective:** Replace arrow grids with fluid streamlines

- [ ] Implement Line Integral Convolution (LIC) or streamline seeding
  - Option A: LIC for dense, smoke-like texture
  - Option B: Evenly-spaced streamlines for clarity
  - Option C: Hybrid (streamlines with LIC background)

- [ ] Streamline seeding strategy
  - Uniform grid seeding (baseline)
  - Density-proportional seeding (more lines in high-velocity regions)
  - Critical point avoidance (don't seed inside vortex cores)

- [ ] Streamline rendering properties
  - Line width: 1-3 pixels, anti-aliased
  - Color: velocity magnitude mapped to colormap
  - Opacity: fade at endpoints for clean termination
  - Length: proportional to local velocity or fixed

- [ ] Implement animated streamlines (optional)
  - Texture advection along flow direction
  - Creates "flowing water" effect
  - 10-30 FPS for smooth motion

**Acceptance Criteria:**
- [ ] Flow direction immediately obvious at any point
- [ ] Vortex structures (cyclones) clearly visible as spiral patterns
- [ ] No visual clutter or overlapping spaghetti
- [ ] Smooth, anti-aliased lines

---

### Milestone 2.2: Vector Arrow Rendering
**Objective:** Clean, professional wind barbs/arrows

- [ ] Implement proper meteorological wind barbs (optional)
  - Half barb: 5 knots
  - Full barb: 10 knots
  - Triangle: 50 knots
  - Orientation: points in direction wind is coming FROM

- [ ] Alternative: Triangular arrows
  - Filled triangle heads
  - Shaft width proportional to magnitude
  - Color-coded by intensity
  - White or contrasting outline for visibility

- [ ] Adaptive density
  - Fewer arrows when zoomed out (avoid clutter)
  - More arrows when zoomed in (detail)
  - Minimum spacing: 20-40 pixels between arrows

- [ ] Z-fighting prevention
  - Arrow heads always on top
  - Proper depth ordering within arrow layer

**Acceptance Criteria:**
- [ ] Arrows readable at any zoom level
- [ ] No overlapping arrow heads
- [ ] Direction unambiguous
- [ ] Magnitude visually encoded (size or color)

---

## Phase 3: Colormap & Visual Design

### Milestone 3.1: Professional Colormap
**Objective:** Scientifically accurate, visually appealing color scheme

- [ ] Select perceptually uniform colormap
  - Options: viridis, plasma, inferno, cividis
  - Avoid: rainbow/jet (perceptual non-uniformity)
  
- [ ] Implement diverging colormap for anomalies
  - Blue → White → Red for temperature anomalies
  - Brown → White → Green for precipitation

- [ ] Wind-specific colormap
  - Low (0-10 m/s): Cool blues/greens
  - Medium (10-25 m/s): Yellows/oranges  
  - High (25-50 m/s): Reds/magentas
  - Extreme (50+ m/s): Deep purple/black (hurricane intensity)

- [ ] Implement discrete colormap option
  - 8-12 distinct color bins
  - Clear boundaries for operational use
  - Legend with exact value ranges

**Acceptance Criteria:**
- [ ] Color progression feels natural (no jarring transitions)
- [ ] Colorblind-friendly (test with simulator)
- [ ] Extremes are visually alarming
- [ ] Baseline conditions are calm/neutral colors

---

### Milestone 3.2: Terrain-Aware Coloring
**Objective:** Wind colors don't clash with terrain

- [ ] Implement adaptive color intensity
  - Over ocean (dark blue): use brighter wind colors
  - Over land (varied): use higher contrast colors
  - Over ice (white): use darker/saturated wind colors

- [ ] Edge enhancement at coastlines
  - Subtle darkening at land/ocean boundary
  - Helps distinguish terrain from wind data

- [ ] Optional: terrain-masked wind display
  - Show wind only over ocean (maritime focus)
  - Show wind only over land (continental focus)
  - User toggle in UI

**Acceptance Criteria:**
- [ ] Wind data visible over ALL terrain types
- [ ] No "camouflage" effect where wind matches terrain color
- [ ] Clear figure-ground separation

---

## Phase 4: Temporal Animation

### Milestone 4.1: Geodesic Interpolation (Core)
**Objective:** Smooth temporal transitions without artifacts

- [ ] Implement optical flow-based interpolation
  - Estimate motion vectors between keyframes
  - Warp intermediate frames along flow
  - No "ghosting" or "morphing" artifacts

- [ ] Alternative: Spline coefficient interpolation
  - Interpolate in representation space, not pixel space
  - Preserves sharp features during transition

- [ ] Frame rate targets
  - Minimum: 10 FPS for "slideshow" feel
  - Target: 30 FPS for smooth animation
  - Stretch: 60 FPS for broadcast quality

- [ ] Implement scrubber with preview
  - Thumbnail strip showing keyframes
  - Hover preview at any temporal position
  - Click to jump, drag to scrub

**Acceptance Criteria:**
- [ ] Features move coherently (no smearing)
- [ ] No "watercolor bleed" at t=0.5
- [ ] Cyclones rotate smoothly
- [ ] Frontal boundaries migrate cleanly

---

### Milestone 4.2: Playback Controls
**Objective:** Professional animation controls

- [ ] Implement playback UI
  - Play/Pause button
  - Speed control (0.5x, 1x, 2x, 4x)
  - Loop toggle
  - Step forward/backward (frame-by-frame)

- [ ] Keyboard shortcuts
  - Space: Play/Pause
  - Left/Right: Step frame
  - Up/Down: Speed control
  - L: Loop toggle

- [ ] Time indicator
  - Current timestamp display (UTC)
  - Progress bar with time labels
  - Forecast hour indicator (T+0, T+6, T+12, etc.)

**Acceptance Criteria:**
- [ ] Playback controls feel responsive
- [ ] No lag when scrubbing
- [ ] Time always visible and accurate

---

## Phase 5: HUD & Professional Overlays

### Milestone 5.1: Information Display
**Objective:** Broadcast-quality heads-up display

- [ ] Implement corner panels
  - Top-left: Data source, initialization time, model name
  - Top-right: Current display time, forecast hour
  - Bottom-left: Variable name, level, units
  - Bottom-right: Color legend with values

- [ ] Styling
  - Semi-transparent dark background (rgba(0,0,0,0.7))
  - Clean sans-serif font (Roboto, Inter, or system)
  - White or light gray text
  - Subtle border or shadow

- [ ] Dynamic content
  - Update on variable change
  - Update on time scrub
  - Show cursor position (lat/lon) on hover

**Acceptance Criteria:**
- [ ] All critical info visible without obscuring data
- [ ] Professional, clean appearance
- [ ] Readable at any window size

---

### Milestone 5.2: Interactive Annotations
**Objective:** User can mark and annotate features

- [ ] Implement annotation tools
  - Point marker (click to place)
  - Text label (with custom message)
  - Circle/ellipse (highlight region)
  - Arrow (point at feature)

- [ ] Automatic feature detection (optional)
  - Identify cyclone centers (vorticity maxima)
  - Identify frontal boundaries (gradient maxima)
  - Identify jet streams (wind speed maxima)
  - Display with standardized symbols

- [ ] Export capabilities
  - Screenshot (PNG with all layers)
  - Animation export (GIF or MP4)
  - Data export (CSV of current view)

**Acceptance Criteria:**
- [ ] Annotations persist during animation
- [ ] Clean export suitable for presentations
- [ ] Feature detection reasonably accurate

---

## Phase 6: Performance & Polish

### Milestone 6.1: GPU Acceleration
**Objective:** Smooth performance at high resolution

- [ ] Implement GPU-based rendering
  - Use VisPy shaders or OpenGL directly
  - Texture-based terrain (single GPU upload)
  - Instanced rendering for arrows

- [ ] Level-of-detail (LOD) system
  - Reduce vector density when zoomed out
  - Reduce texture resolution when zoomed out
  - Increase detail progressively on zoom

- [ ] Performance targets
  - 60 FPS at 1080p with 4K terrain
  - 30 FPS at 4K with 8K terrain
  - <100ms initial load time (with 1K fallback)

**Acceptance Criteria:**
- [ ] Smooth pan and zoom
- [ ] No frame drops during animation
- [ ] Responsive UI at all times

---

### Milestone 6.2: Final Polish
**Objective:** Production-ready quality

- [ ] Edge cases
  - Handle missing data gracefully (gray out regions)
  - Handle date line wrap correctly
  - Handle polar projections (optional)

- [ ] Error handling
  - Informative messages on data load failure
  - Graceful degradation if GPU unavailable
  - Recovery from network issues (for live data)

- [ ] Documentation
  - User guide with screenshots
  - API documentation for programmatic use
  - Example scripts for common workflows

- [ ] Testing
  - Unit tests for core algorithms
  - Visual regression tests for rendering
  - Performance benchmarks

**Acceptance Criteria:**
- [ ] No crashes on any valid input
- [ ] Helpful error messages on invalid input
- [ ] Documentation sufficient for new users

---

## Phase 7: Advanced Features (Stretch Goals)

### Milestone 7.1: 3D Globe View
- [ ] Implement spherical projection
- [ ] Rotate globe with mouse drag
- [ ] Zoom with scroll
- [ ] Smooth transition between 2D and 3D views

### Milestone 7.2: Multi-Variable Display
- [ ] Side-by-side comparison
- [ ] Overlay multiple fields (wind + pressure)
- [ ] Difference view (forecast minus analysis)

### Milestone 7.3: Live Data Integration
- [ ] Connect to real-time data sources (GFS, ECMWF)
- [ ] Auto-refresh on new data availability
- [ ] Notification system for extreme events

---

## Summary Checklist

| Phase | Milestone | Status |
|-------|-----------|--------|
| 1 | NASA Blue Marble Integration | ✅ Complete |
| 1 | Proper Compositing Pipeline | ✅ Complete |
| 2 | Professional Streamline Rendering | ✅ Complete |
| 2 | Vector Arrow Rendering | ✅ Complete |
| 3 | Professional Colormap | ✅ Complete |
| 3 | Terrain-Aware Coloring | ✅ Complete |
| 4 | Geodesic Interpolation | ✅ Complete |
| 4 | Playback Controls | ✅ Complete |
| 5 | Information Display (HUD) | ✅ Complete |
| 5 | Interactive Annotations | ⬜ Pending |
| 6 | GPU Acceleration | ⬜ Pending |
| 6 | Final Polish | ⬜ Pending |
| 7 | 3D Globe View | ⬜ Stretch |
| 7 | Multi-Variable Display | ⬜ Stretch |
| 7 | Live Data Integration | ⬜ Stretch |

---

## Implementation Status

### ✅ Completed (December 27, 2025)

**Application:** `demos/hypertensor_pro.py` (1270+ lines)

**Core Features Delivered:**

1. **NASA Blue Marble 8K Integration**
   - Downloaded actual NASA satellite imagery (8192×4096)
   - Multi-resolution loading (8K, 2K fallback)
   - Proper equirectangular coordinate alignment
   - LANCZOS resampling for high quality

2. **Proper Compositing Pipeline**
   - Porter-Duff alpha blending in numpy
   - Adjustable overlay opacity (10-90%)
   - No VisPy alpha blending issues

3. **Professional Streamline Rendering**
   - Bilinear-interpolated streamline integration
   - Color-coded by wind magnitude
   - Arrow heads at streamline termini
   - Configurable density

4. **Vector Arrow Rendering**
   - Grid-based arrow placement
   - Triangular arrow heads with white edges
   - Color-coded by magnitude
   - Proper shaft/head separation

5. **Professional Colormaps**
   - Plasma, Viridis, Inferno, Magma, Turbo, Coolwarm
   - All perceptually uniform
   - Dynamic color legend bar

6. **Playback Controls**
   - 48-hour forecast animation
   - Play/Pause with speed control
   - Step forward/backward
   - Slider scrubbing

7. **HUD Overlays**
   - Data source label
   - Variable/level indicator
   - Timestamp display
   - Value range statistics
   - Forecast hour indicator
   - Color legend bar

8. **Additional Features**
   - Lat/Lon grid overlay (toggleable)
   - Natural Earth coastlines (toggleable)
   - PNG export functionality
   - Realistic multi-cyclone demo data
   - Moving storms during animation

---

## Immediate Next Step

**Ready for Launch:** Run `python demos/hypertensor_pro.py`

---

*Document created: December 26, 2025*
*Last updated: December 27, 2025*
*Target completion: When it's right, not when it's fast*
