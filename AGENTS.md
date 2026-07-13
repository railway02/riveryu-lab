# RiverYu's Lab

## Project

This repository contains a Hugo website using the PaperMod theme.
The site combines a research portfolio, technical writing, projects,
and a connected knowledge garden.

## Product goal

The website should help visitors quickly understand:

1. Who RiverYu is.
2. What research questions he is working on.
3. What projects and results he has completed.
4. How his articles and notes are connected.

## Target information architecture

Top navigation:

- Research
- Projects
- Knowledge
- About
- Search

Knowledge contains:

- Topics
- Articles
- Notes
- Knowledge Map
- Archive

## Content model

- Project: work that demonstrates an actual contribution, experiment or result.
- Article: complete and independently readable long-form writing.
- Note: atomic, evolving or reference-oriented knowledge.
- Topic: a curated hub that connects projects, articles and notes.

## Graph principles

- Do not implement Graph View before internal links and backlinks work.
- Graph edges should primarily come from explicit semantic links.
- Do not connect every page merely because it shares a broad tag.
- Exclude search, archives, pagination and utility pages from the graph.

## Engineering rules

- Inspect the repository before editing.
- Make small, reversible changes.
- Do not replace the Hugo theme unless explicitly requested.
- Preserve existing public URLs whenever possible.
- Add aliases or redirects when URLs must change.
- Do not add production dependencies without approval.
- Preserve dark mode, mobile responsiveness, formulas and code blocks.
- Never push directly to main.
- Run `hugo --minify` before declaring a task complete.
- Report changed files, validation results and remaining risks.

## Visual principles

- Content and research evidence are the main focus.
- Avoid excessive cards, shadows and rounded corners.
- Keep the existing River/water visual identity.
- Graph View is an exploration tool, not the visual centerpiece.
- Check desktop and mobile layouts after significant UI changes.

