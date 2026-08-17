# Engineering Onboarding Guide

Welcome to Northwind. This guide covers your first two weeks.

## Day one

On your first day, collect your laptop from the IT desk on the 3rd floor. Your
manager will send a calendar invite for a 30-minute welcome sync. Complete the
mandatory security training in the Learning portal before you are granted
production access.

## Accounts and access

Access requests go through the internal portal at access.northwind.internal.
Standard engineers receive read access to the staging environment on day one.
Write access to production requires manager approval and completion of the
on-call readiness checklist. Access reviews happen every quarter; stale grants
are revoked automatically after 90 days of no use.

## Your development environment

We use a standardized dev container. Clone the `platform/dev-env` repository and
run `make bootstrap`. This installs the toolchain, pre-commit hooks, and local
service dependencies. The bootstrap script pins all versions so every engineer
runs an identical environment.

## Code review

Every change ships through a pull request with at least one approval. Reviews
are expected within one business day. Large changes (over 400 lines) should be
split unless there is a documented reason not to. CI must be green before merge;
the pipeline runs unit tests, linting, and the evaluation gate.

## Getting help

Post in the #eng-help channel for anything blocking you. There are no bad
questions in your first month. Your onboarding buddy is available for pairing
sessions any afternoon during your first two weeks.
