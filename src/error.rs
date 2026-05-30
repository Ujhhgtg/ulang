//! Source-level error reporting utilities.
//!
//! This module provides functions for emitting pretty, source-anchored error messages
//! to stderr using the [`annotate_snippets`] crate. Errors are displayed with the
//! offending source range highlighted, a label, and file path context.

use annotate_snippets::renderer::DecorStyle;
use annotate_snippets::{AnnotationKind, Level, Renderer, Snippet};

use crate::token::Span;

/// Emit a source-level error to stderr with the highlighted span.
///
/// Computes a 1-indexed line number from the byte offset in `span.lo`, then formats
/// the error using `annotate_snippets` to show the exact source range. The output
/// includes the file `path`, the error `title`, and a `label` attached to the span.
///
/// Use this when the error location is known and a source-level highlight is desired.
///
/// # Panics
///
/// May panic if `span.lo` or `span.hi` exceed `source.len()`, causing a substring
/// access out of bounds, or if the byte offset falls in the middle of a multi-byte
/// UTF-8 character (the line-number computation indexes into `source[..span.lo]` and
/// will panic on a non-char boundary).
pub fn emit_error(source: &str, path: &str, span: Span, title: &str, label: &str) {
    // Compute 1-indexed line number from byte offset
    let line_start = source[..span.lo].matches('\n').count() + 1;

    let report = Level::ERROR.primary_title(title).element(
        Snippet::source(source)
            .line_start(line_start)
            .fold(false)
            .path(path)
            .annotation(AnnotationKind::Primary.span(span.lo..span.hi).label(label)),
    );

    let renderer = Renderer::styled().decor_style(DecorStyle::Unicode);
    eprintln!("{}", renderer.render(&[report]));
}

/// Emit an error with an optional span.
///
/// If `span` is `Some`, delegates to [`emit_error`] for a source-highlighted diagnostic.
/// If `span` is `None`, falls back to a plain `eprintln!("{}: {}", title, label)` with
/// no source location context.
///
/// Use this when an error may originate from a synthetic element or a parsing phase
/// where a source span is not available.
///
/// # Panics
///
/// Inherits the same panic conditions as [`emit_error`] when `span` is `Some`.
pub fn emit_error_opt(source: &str, path: &str, span: Option<Span>, title: &str, label: &str) {
    match span {
        Some(span) => emit_error(source, path, span, title, label),
        None => eprintln!("{}: {}", title, label),
    }
}
