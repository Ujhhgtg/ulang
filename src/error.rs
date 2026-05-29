use annotate_snippets::renderer::DecorStyle;
use annotate_snippets::{AnnotationKind, Level, Renderer, Snippet};

use crate::token::Span;

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

/// Emit an error with an optional span. If span is Some, uses annotate_snippets for
/// pretty source-level diagnostics. If span is None, falls back to a plain eprintln.
pub fn emit_error_opt(source: &str, path: &str, span: Option<Span>, title: &str, label: &str) {
    match span {
        Some(span) => emit_error(source, path, span, title, label),
        None => eprintln!("{}: {}", title, label),
    }
}
