The plain text builder treats = and ~ as verbatim only when the
characters around them match org rules, and it leaves \( \) and $$
math unchanged. llms-full.txt is checked for a backtick inside inline
math and for an unbalanced backtick on a line.
