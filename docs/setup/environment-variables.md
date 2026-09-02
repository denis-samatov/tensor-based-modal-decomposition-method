# Environment Variables

## Purpose
Documents the environment variables used by the project.

## Audience
Developers configuring their local environment.

## Summary
The TBMD project is a numerical library and does not require extensive environment variable configuration. 

## Details
There are no mandatory environment variables for the core library.

Dataset-specific orchestration may define its own environment variables, but the core `TBMD`
library does not parse an `.env` file. RANS/URANS forecasting configuration belongs to the
separate `tbmd-forecasting` repository.

## Validation
Ensure `.env` files are not tracked by git.
