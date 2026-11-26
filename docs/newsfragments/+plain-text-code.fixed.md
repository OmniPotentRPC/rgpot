The plain text builder copies backtick code through unchanged and
keeps a marker that belongs inside a code span. A slash, a plus or a
star still ends a span, and math is left unchanged. The check reports
an inline code span from the sources that the plain text changed.
