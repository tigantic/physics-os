// VALHALLA Photonic Discipline - Global Shader Constants
// ========================================================
// OPERATION VALHALLA - Phase 4.1
// 
// Tactical color palette hardcoded into GPU pipeline.
// All fragment shaders MUST include this header.

// === 1.1 THE SUBSTRATE ===
#define OBSIDIAN_DEEP vec3(0.039, 0.039, 0.043)  // #0A0A0B
#define VOID_BLACK vec3(0.0, 0.0, 0.0)           // #000000

// === 1.2 PRIMARY DATA ===
#define ISOTOPE_WHITE vec3(0.878, 0.878, 0.878)  // #E0E0E0
#define PURE_WHITE vec3(1.0, 1.0, 1.0)           // #FFFFFF

// === 1.3 FIELD GRADIENTS (Plasma) ===
#define PLASMA_LOW vec3(0.051, 0.031, 0.529)     // #0D0887 Deep Indigo
#define PLASMA_MID_LOW vec3(0.416, 0.0, 0.659)   // #6A00A8 Violet
#define PLASMA_MID vec3(0.694, 0.165, 0.565)     // #B12A90 Magenta
#define PLASMA_MID_HIGH vec3(0.882, 0.392, 0.384) // #E16462 Coral
#define PLASMA_HIGH vec3(0.988, 0.650, 0.212)    // #FCA636 Radon Amber

// === 1.4 ACCENTS ===
#define RADON_AMBER vec3(1.0, 0.702, 0.0)        // #FFB300
#define CYGNUS_BLUE vec3(0.0, 0.898, 1.0)        // #00E5FF

// === 1.5 GHOST LAYER ===
#define GHOST_SLATE vec3(0.184, 0.204, 0.247)    // #2F343F

// === STATUS COLORS ===
#define DANGER_RED vec3(1.0, 0.231, 0.188)       // #FF3B30
#define SUCCESS_GREEN vec3(0.204, 0.780, 0.349)  // #34C759
#define WARNING_ORANGE vec3(1.0, 0.584, 0.0)     // #FF9500

// === OPACITY MAPPING ===
// Signal burns through noise: low values near-invisible, anomalies burn bright
float opacity_map(float value, float min_alpha, float max_alpha) {
    return min_alpha + value * (max_alpha - min_alpha);
}

// === PLASMA GRADIENT ===
// Perceptually uniform color mapping for scalar fields
vec3 plasma_gradient(float t) {
    t = clamp(t, 0.0, 1.0);
    
    if (t < 0.25) {
        float local_t = t / 0.25;
        return mix(PLASMA_LOW, PLASMA_MID_LOW, local_t);
    } else if (t < 0.5) {
        float local_t = (t - 0.25) / 0.25;
        return mix(PLASMA_MID_LOW, PLASMA_MID, local_t);
    } else if (t < 0.75) {
        float local_t = (t - 0.5) / 0.25;
        return mix(PLASMA_MID, PLASMA_MID_HIGH, local_t);
    } else {
        float local_t = (t - 0.75) / 0.25;
        return mix(PLASMA_MID_HIGH, PLASMA_HIGH, local_t);
    }
}
