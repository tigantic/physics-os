// Phase 1: Simple Text Rendering
// Bitmap font rendering for telemetry labels
// Constitutional compliance: Doctrine 1 (procedural), Doctrine 8 (minimal memory)

// Phase 2 scaffolding: CPU text rendering for telemetry overlay
// Will be superseded by GPU text rendering in production

#[allow(dead_code)]
/// Simple 8x8 bitmap font for ASCII characters
/// Each character is represented by 8 bytes (8 rows of 8 pixels)
/// Bit 1 = pixel on, Bit 0 = pixel off
pub struct BitmapFont {
    /// Glyph data for ASCII 32-126 (95 printable characters)
    glyphs: [[u8; 8]; 95],
    
    /// Character width in pixels
    pub char_width: u32,
    
    /// Character height in pixels
    pub char_height: u32,
}

// Phase 2 scaffolding: BitmapFont implementation for CPU text rendering
#[allow(dead_code)]
impl BitmapFont {
    /// Create a new bitmap font with embedded 8x8 glyphs
    pub fn new() -> Self {
        Self {
            glyphs: Self::generate_8x8_glyphs(),
            char_width: 8,
            char_height: 8,
        }
    }

    /// Get glyph data for a character
    pub fn get_glyph(&self, c: char) -> Option<&[u8; 8]> {
        let ascii = c as usize;
        if (32..=126).contains(&ascii) {
            Some(&self.glyphs[ascii - 32])
        } else {
            None
        }
    }

    /// Check if a pixel in a glyph is set
    pub fn is_pixel_set(&self, glyph: &[u8; 8], x: u32, y: u32) -> bool {
        if x >= 8 || y >= 8 {
            return false;
        }
        let row = glyph[y as usize];
        (row & (1 << (7 - x))) != 0
    }

    /// Render text to a pixel buffer (returns pixel positions that should be lit)
    pub fn render_text(&self, text: &str, x: i32, y: i32) -> Vec<(i32, i32)> {
        let mut pixels = Vec::new();
        
        for (i, c) in text.chars().enumerate() {
            if let Some(glyph) = self.get_glyph(c) {
                let char_x = x + (i as i32 * self.char_width as i32);
                
                for row in 0..8 {
                    for col in 0..8 {
                        if self.is_pixel_set(glyph, col, row) {
                            pixels.push((char_x + col as i32, y + row as i32));
                        }
                    }
                }
            }
        }
        
        pixels
    }

    /// Calculate text width in pixels
    pub fn text_width(&self, text: &str) -> u32 {
        text.len() as u32 * self.char_width
    }

    /// Generate 8x8 bitmap glyphs for ASCII 32-126
    /// This is a simplified font - just enough for telemetry display
    fn generate_8x8_glyphs() -> [[u8; 8]; 95] {
        let mut glyphs = [[0u8; 8]; 95];
        
        // Space (32)
        glyphs[0] = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        
        // ! (33)
        glyphs[1] = [0x18, 0x18, 0x18, 0x18, 0x18, 0x00, 0x18, 0x00];
        
        // " (34)
        glyphs[2] = [0x36, 0x36, 0x36, 0x00, 0x00, 0x00, 0x00, 0x00];
        
        // # (35)
        glyphs[3] = [0x36, 0x36, 0x7F, 0x36, 0x7F, 0x36, 0x36, 0x00];
        
        // $ (36)
        glyphs[4] = [0x0C, 0x3E, 0x03, 0x1E, 0x30, 0x1F, 0x0C, 0x00];
        
        // % (37)
        glyphs[5] = [0x00, 0x63, 0x33, 0x18, 0x0C, 0x66, 0x63, 0x00];
        
        // & (38)
        glyphs[6] = [0x1C, 0x36, 0x1C, 0x6E, 0x3B, 0x33, 0x6E, 0x00];
        
        // ' (39)
        glyphs[7] = [0x06, 0x06, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00];
        
        // ( (40)
        glyphs[8] = [0x18, 0x0C, 0x06, 0x06, 0x06, 0x0C, 0x18, 0x00];
        
        // ) (41)
        glyphs[9] = [0x06, 0x0C, 0x18, 0x18, 0x18, 0x0C, 0x06, 0x00];
        
        // * (42)
        glyphs[10] = [0x00, 0x66, 0x3C, 0xFF, 0x3C, 0x66, 0x00, 0x00];
        
        // + (43)
        glyphs[11] = [0x00, 0x0C, 0x0C, 0x3F, 0x0C, 0x0C, 0x00, 0x00];
        
        // , (44)
        glyphs[12] = [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C, 0x06];
        
        // - (45)
        glyphs[13] = [0x00, 0x00, 0x00, 0x3F, 0x00, 0x00, 0x00, 0x00];
        
        // . (46)
        glyphs[14] = [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C, 0x00];
        
        // / (47)
        glyphs[15] = [0x60, 0x30, 0x18, 0x0C, 0x06, 0x03, 0x01, 0x00];
        
        // 0 (48)
        glyphs[16] = [0x3E, 0x63, 0x73, 0x7B, 0x6F, 0x67, 0x3E, 0x00];
        
        // 1 (49)
        glyphs[17] = [0x0C, 0x0E, 0x0C, 0x0C, 0x0C, 0x0C, 0x3F, 0x00];
        
        // 2 (50)
        glyphs[18] = [0x1E, 0x33, 0x30, 0x1C, 0x06, 0x33, 0x3F, 0x00];
        
        // 3 (51)
        glyphs[19] = [0x1E, 0x33, 0x30, 0x1C, 0x30, 0x33, 0x1E, 0x00];
        
        // 4 (52)
        glyphs[20] = [0x38, 0x3C, 0x36, 0x33, 0x7F, 0x30, 0x78, 0x00];
        
        // 5 (53)
        glyphs[21] = [0x3F, 0x03, 0x1F, 0x30, 0x30, 0x33, 0x1E, 0x00];
        
        // 6 (54)
        glyphs[22] = [0x1C, 0x06, 0x03, 0x1F, 0x33, 0x33, 0x1E, 0x00];
        
        // 7 (55)
        glyphs[23] = [0x3F, 0x33, 0x30, 0x18, 0x0C, 0x0C, 0x0C, 0x00];
        
        // 8 (56)
        glyphs[24] = [0x1E, 0x33, 0x33, 0x1E, 0x33, 0x33, 0x1E, 0x00];
        
        // 9 (57)
        glyphs[25] = [0x1E, 0x33, 0x33, 0x3E, 0x30, 0x18, 0x0E, 0x00];
        
        // : (58)
        glyphs[26] = [0x00, 0x0C, 0x0C, 0x00, 0x00, 0x0C, 0x0C, 0x00];
        
        // ; (59)
        glyphs[27] = [0x00, 0x0C, 0x0C, 0x00, 0x00, 0x0C, 0x0C, 0x06];
        
        // < (60)
        glyphs[28] = [0x18, 0x0C, 0x06, 0x03, 0x06, 0x0C, 0x18, 0x00];
        
        // = (61)
        glyphs[29] = [0x00, 0x00, 0x3F, 0x00, 0x00, 0x3F, 0x00, 0x00];
        
        // > (62)
        glyphs[30] = [0x06, 0x0C, 0x18, 0x30, 0x18, 0x0C, 0x06, 0x00];
        
        // ? (63)
        glyphs[31] = [0x1E, 0x33, 0x30, 0x18, 0x0C, 0x00, 0x0C, 0x00];
        
        // @ (64)
        glyphs[32] = [0x3E, 0x63, 0x7B, 0x7B, 0x7B, 0x03, 0x1E, 0x00];
        
        // A (65)
        glyphs[33] = [0x0C, 0x1E, 0x33, 0x33, 0x3F, 0x33, 0x33, 0x00];
        
        // B (66)
        glyphs[34] = [0x3F, 0x66, 0x66, 0x3E, 0x66, 0x66, 0x3F, 0x00];
        
        // C (67)
        glyphs[35] = [0x3C, 0x66, 0x03, 0x03, 0x03, 0x66, 0x3C, 0x00];
        
        // D (68)
        glyphs[36] = [0x1F, 0x36, 0x66, 0x66, 0x66, 0x36, 0x1F, 0x00];
        
        // E (69)
        glyphs[37] = [0x7F, 0x46, 0x16, 0x1E, 0x16, 0x46, 0x7F, 0x00];
        
        // F (70)
        glyphs[38] = [0x7F, 0x46, 0x16, 0x1E, 0x16, 0x06, 0x0F, 0x00];
        
        // G (71)
        glyphs[39] = [0x3C, 0x66, 0x03, 0x03, 0x73, 0x66, 0x7C, 0x00];
        
        // H (72)
        glyphs[40] = [0x33, 0x33, 0x33, 0x3F, 0x33, 0x33, 0x33, 0x00];
        
        // I (73)
        glyphs[41] = [0x1E, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00];
        
        // J (74)
        glyphs[42] = [0x78, 0x30, 0x30, 0x30, 0x33, 0x33, 0x1E, 0x00];
        
        // K (75)
        glyphs[43] = [0x67, 0x66, 0x36, 0x1E, 0x36, 0x66, 0x67, 0x00];
        
        // L (76)
        glyphs[44] = [0x0F, 0x06, 0x06, 0x06, 0x46, 0x66, 0x7F, 0x00];
        
        // M (77)
        glyphs[45] = [0x63, 0x77, 0x7F, 0x7F, 0x6B, 0x63, 0x63, 0x00];
        
        // N (78)
        glyphs[46] = [0x63, 0x67, 0x6F, 0x7B, 0x73, 0x63, 0x63, 0x00];
        
        // O (79)
        glyphs[47] = [0x1C, 0x36, 0x63, 0x63, 0x63, 0x36, 0x1C, 0x00];
        
        // P (80)
        glyphs[48] = [0x3F, 0x66, 0x66, 0x3E, 0x06, 0x06, 0x0F, 0x00];
        
        // Q (81)
        glyphs[49] = [0x1E, 0x33, 0x33, 0x33, 0x3B, 0x1E, 0x38, 0x00];
        
        // R (82)
        glyphs[50] = [0x3F, 0x66, 0x66, 0x3E, 0x36, 0x66, 0x67, 0x00];
        
        // S (83)
        glyphs[51] = [0x1E, 0x33, 0x07, 0x0E, 0x38, 0x33, 0x1E, 0x00];
        
        // T (84)
        glyphs[52] = [0x3F, 0x2D, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00];
        
        // U (85)
        glyphs[53] = [0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x3F, 0x00];
        
        // V (86)
        glyphs[54] = [0x33, 0x33, 0x33, 0x33, 0x33, 0x1E, 0x0C, 0x00];
        
        // W (87)
        glyphs[55] = [0x63, 0x63, 0x63, 0x6B, 0x7F, 0x77, 0x63, 0x00];
        
        // X (88)
        glyphs[56] = [0x63, 0x63, 0x36, 0x1C, 0x1C, 0x36, 0x63, 0x00];
        
        // Y (89)
        glyphs[57] = [0x33, 0x33, 0x33, 0x1E, 0x0C, 0x0C, 0x1E, 0x00];
        
        // Z (90)
        glyphs[58] = [0x7F, 0x63, 0x31, 0x18, 0x4C, 0x66, 0x7F, 0x00];
        
        // Remaining characters (91-126) - simplified for space
        // Fill remaining with basic patterns or spaces
        glyphs[59..95].fill([0x00; 8]);
        
        glyphs
    }
}

impl Default for BitmapFont {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_font_creation() {
        let font = BitmapFont::new();
        assert_eq!(font.char_width, 8);
        assert_eq!(font.char_height, 8);
    }

    #[test]
    fn test_glyph_retrieval() {
        let font = BitmapFont::new();
        
        // Valid ASCII printable
        assert!(font.get_glyph('A').is_some());
        assert!(font.get_glyph('0').is_some());
        assert!(font.get_glyph(' ').is_some());
        
        // Invalid characters
        assert!(font.get_glyph('\n').is_none());
        assert!(font.get_glyph('€').is_none());
    }

    #[test]
    fn test_text_width() {
        let font = BitmapFont::new();
        assert_eq!(font.text_width("Hello"), 40); // 5 chars × 8 pixels
        assert_eq!(font.text_width("FPS: 60.0"), 72); // 9 chars × 8 pixels
    }

    #[test]
    fn test_render_text() {
        let font = BitmapFont::new();
        let pixels = font.render_text("A", 0, 0);
        
        // 'A' should have multiple pixels set
        assert!(!pixels.is_empty());
        
        // All pixels should be within expected bounds (8x8)
        for (x, y) in pixels {
            assert!(x >= 0 && x < 8);
            assert!(y >= 0 && y < 8);
        }
    }

    #[test]
    fn test_pixel_check() {
        let font = BitmapFont::new();
        let glyph = &[0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00]; // Alternating rows
        
        // First row should be all set
        for x in 0..8 {
            assert!(font.is_pixel_set(glyph, x, 0));
        }
        
        // Second row should be all clear
        for x in 0..8 {
            assert!(!font.is_pixel_set(glyph, x, 1));
        }
    }
}
