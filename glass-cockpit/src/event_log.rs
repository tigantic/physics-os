/*!
 * Phase 7: Event Log Terminal
 * 
 * Bottom pane scrolling event log with:
 * - Timestamped HyperTensor events
 * - Filtering by event type
 * - Color-coded severity levels
 * - High-throughput logging without lag
 */
#![allow(dead_code)] // Event log API ready for integration

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Maximum events to keep in memory
const MAX_EVENTS: usize = 1000;

/// Event severity levels
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventLevel {
    Debug,
    Info,
    Warning,
    Error,
    Critical,
}

impl EventLevel {
    pub fn color(&self) -> [f32; 4] {
        match self {
            EventLevel::Debug => [0.5, 0.5, 0.5, 1.0],    // Gray
            EventLevel::Info => [0.7, 0.9, 1.0, 1.0],     // Light blue
            EventLevel::Warning => [1.0, 0.8, 0.2, 1.0],  // Yellow
            EventLevel::Error => [1.0, 0.3, 0.3, 1.0],    // Red
            EventLevel::Critical => [1.0, 0.1, 0.5, 1.0], // Magenta
        }
    }
    
    pub fn prefix(&self) -> &'static str {
        match self {
            EventLevel::Debug => "[DBG]",
            EventLevel::Info => "[INF]",
            EventLevel::Warning => "[WRN]",
            EventLevel::Error => "[ERR]",
            EventLevel::Critical => "[CRT]",
        }
    }
}

/// Event categories for filtering
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventCategory {
    System,     // CPU, memory, general system
    Physics,    // Fluid dynamics, field updates
    Render,     // GPU, frame timing, rendering
    Bridge,     // RAM bridge, Python↔Rust communication
    Input,      // User input, keyboard/mouse
    Network,    // If applicable
}

impl EventCategory {
    pub fn tag(&self) -> &'static str {
        match self {
            EventCategory::System => "SYS",
            EventCategory::Physics => "PHY",
            EventCategory::Render => "RND",
            EventCategory::Bridge => "BRG",
            EventCategory::Input => "INP",
            EventCategory::Network => "NET",
        }
    }
}

/// Single log event
#[derive(Clone)]
pub struct LogEvent {
    pub timestamp: Duration,    // Time since app start
    pub level: EventLevel,
    pub category: EventCategory,
    pub message: String,
}

impl LogEvent {
    pub fn formatted(&self) -> String {
        let secs = self.timestamp.as_secs();
        let millis = self.timestamp.subsec_millis();
        format!(
            "{:02}:{:02}.{:03} {} [{}] {}",
            secs / 60,
            secs % 60,
            millis,
            self.level.prefix(),
            self.category.tag(),
            self.message
        )
    }
}

/// Filter configuration
#[derive(Clone)]
pub struct EventFilter {
    pub min_level: EventLevel,
    pub categories: Vec<EventCategory>,
    pub search_text: Option<String>,
}

impl Default for EventFilter {
    fn default() -> Self {
        Self {
            min_level: EventLevel::Debug,
            categories: vec![
                EventCategory::System,
                EventCategory::Physics,
                EventCategory::Render,
                EventCategory::Bridge,
                EventCategory::Input,
                EventCategory::Network,
            ],
            search_text: None,
        }
    }
}

impl EventFilter {
    pub fn matches(&self, event: &LogEvent) -> bool {
        // Check level
        let level_order = |l: EventLevel| match l {
            EventLevel::Debug => 0,
            EventLevel::Info => 1,
            EventLevel::Warning => 2,
            EventLevel::Error => 3,
            EventLevel::Critical => 4,
        };
        
        if level_order(event.level) < level_order(self.min_level) {
            return false;
        }
        
        // Check category
        if !self.categories.contains(&event.category) {
            return false;
        }
        
        // Check search text
        if let Some(ref search) = self.search_text {
            if !event.message.to_lowercase().contains(&search.to_lowercase()) {
                return false;
            }
        }
        
        true
    }
}

/// Event log manager
pub struct EventLog {
    events: VecDeque<LogEvent>,
    start_time: Instant,
    filter: EventFilter,
    scroll_offset: usize,
    auto_scroll: bool,
    visible_lines: usize,
}

impl EventLog {
    pub fn new() -> Self {
        Self {
            events: VecDeque::with_capacity(MAX_EVENTS),
            start_time: Instant::now(),
            filter: EventFilter::default(),
            scroll_offset: 0,
            auto_scroll: true,
            visible_lines: 10,
        }
    }
    
    /// Log a new event
    pub fn log(&mut self, level: EventLevel, category: EventCategory, message: impl Into<String>) {
        let event = LogEvent {
            timestamp: self.start_time.elapsed(),
            level,
            category,
            message: message.into(),
        };
        
        if self.events.len() >= MAX_EVENTS {
            self.events.pop_front();
        }
        
        self.events.push_back(event);
        
        // Auto-scroll to bottom
        if self.auto_scroll {
            let filtered_count = self.filtered_events().count();
            self.scroll_offset = filtered_count.saturating_sub(self.visible_lines);
        }
    }
    
    /// Convenience methods
    pub fn debug(&mut self, category: EventCategory, message: impl Into<String>) {
        self.log(EventLevel::Debug, category, message);
    }
    
    pub fn info(&mut self, category: EventCategory, message: impl Into<String>) {
        self.log(EventLevel::Info, category, message);
    }
    
    pub fn warn(&mut self, category: EventCategory, message: impl Into<String>) {
        self.log(EventLevel::Warning, category, message);
    }
    
    pub fn error(&mut self, category: EventCategory, message: impl Into<String>) {
        self.log(EventLevel::Error, category, message);
    }
    
    pub fn critical(&mut self, category: EventCategory, message: impl Into<String>) {
        self.log(EventLevel::Critical, category, message);
    }
    
    /// Get filtered events iterator
    pub fn filtered_events(&self) -> impl Iterator<Item = &LogEvent> {
        self.events.iter().filter(|e| self.filter.matches(e))
    }
    
    /// Get visible events for rendering
    pub fn visible_events(&self) -> Vec<&LogEvent> {
        self.filtered_events()
            .skip(self.scroll_offset)
            .take(self.visible_lines)
            .collect()
    }
    
    /// Set filter
    pub fn set_filter(&mut self, filter: EventFilter) {
        self.filter = filter;
        // Reset scroll when filter changes
        self.scroll_offset = 0;
    }
    
    /// Set minimum log level
    pub fn set_min_level(&mut self, level: EventLevel) {
        self.filter.min_level = level;
    }
    
    /// Toggle category filter
    pub fn toggle_category(&mut self, category: EventCategory) {
        if let Some(pos) = self.filter.categories.iter().position(|&c| c == category) {
            self.filter.categories.remove(pos);
        } else {
            self.filter.categories.push(category);
        }
    }
    
    /// Set search text
    pub fn set_search(&mut self, text: Option<String>) {
        self.filter.search_text = text;
    }
    
    /// Scroll up
    pub fn scroll_up(&mut self, lines: usize) {
        self.scroll_offset = self.scroll_offset.saturating_sub(lines);
        self.auto_scroll = false;
    }
    
    /// Scroll down
    pub fn scroll_down(&mut self, lines: usize) {
        let max_offset = self.filtered_events().count().saturating_sub(self.visible_lines);
        self.scroll_offset = (self.scroll_offset + lines).min(max_offset);
        
        // Re-enable auto-scroll if at bottom
        if self.scroll_offset >= max_offset {
            self.auto_scroll = true;
        }
    }
    
    /// Scroll to bottom
    pub fn scroll_to_bottom(&mut self) {
        let max_offset = self.filtered_events().count().saturating_sub(self.visible_lines);
        self.scroll_offset = max_offset;
        self.auto_scroll = true;
    }
    
    /// Set visible lines (based on terminal height)
    pub fn set_visible_lines(&mut self, lines: usize) {
        self.visible_lines = lines.max(1);
    }
    
    /// Get event count
    pub fn event_count(&self) -> usize {
        self.events.len()
    }
    
    /// Get filtered event count
    pub fn filtered_count(&self) -> usize {
        self.filtered_events().count()
    }
    
    /// Clear all events
    pub fn clear(&mut self) {
        self.events.clear();
        self.scroll_offset = 0;
    }
    
    /// Export events to string
    pub fn export(&self) -> String {
        self.events
            .iter()
            .map(|e| e.formatted())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Thread-safe event logger (for async logging)
pub struct AsyncEventLogger {
    sender: std::sync::mpsc::Sender<LogEvent>,
}

impl AsyncEventLogger {
    pub fn new(sender: std::sync::mpsc::Sender<LogEvent>) -> Self {
        Self { sender }
    }
    
    pub fn log(&self, level: EventLevel, category: EventCategory, message: impl Into<String>) {
        let event = LogEvent {
            timestamp: Duration::from_secs(0), // Will be set by receiver
            level,
            category,
            message: message.into(),
        };
        let _ = self.sender.send(event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_event_log_basic() {
        let mut log = EventLog::new();
        
        log.info(EventCategory::System, "Application started");
        log.debug(EventCategory::Render, "Frame rendered");
        log.warn(EventCategory::Physics, "High vorticity detected");
        log.error(EventCategory::Bridge, "Connection lost");
        
        assert_eq!(log.event_count(), 4);
    }
    
    #[test]
    fn test_event_filter() {
        let mut log = EventLog::new();
        
        log.debug(EventCategory::Render, "Debug message");
        log.info(EventCategory::Render, "Info message");
        log.warn(EventCategory::Render, "Warning message");
        log.error(EventCategory::Render, "Error message");
        
        log.set_min_level(EventLevel::Warning);
        
        assert_eq!(log.filtered_count(), 2);
    }
    
    #[test]
    fn test_category_filter() {
        let mut log = EventLog::new();
        
        log.info(EventCategory::System, "System event");
        log.info(EventCategory::Render, "Render event");
        log.info(EventCategory::Physics, "Physics event");
        
        log.toggle_category(EventCategory::Render);
        
        assert_eq!(log.filtered_count(), 2);
    }
    
    #[test]
    fn test_max_events() {
        let mut log = EventLog::new();
        
        for i in 0..1500 {
            log.info(EventCategory::System, format!("Event {}", i));
        }
        
        assert_eq!(log.event_count(), MAX_EVENTS);
    }
}
