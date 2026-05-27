# Proactive Forgetting-Risk Estimation for Replay Memory Selection

🚧 🚧 *The methodology and findings are yet to be well documented. My apologies.* 🚧 🚧

This repository contains a proof of concept for testing whether proactive forgetting-risk estimation can guide replay memory selection in continual learning.

The experiment studies sample-level forgetting risk before extending the idea toward token/subtoken-level memory in multimodal continual learning.

## Core Question

Can we estimate which old samples are likely to be forgotten before training on a new task, and use this estimate to improve replay memory selection?

## Short Finding

Forgetting-risk estimation is viable as a diagnostic signal, but risk-only replay did not beat random replay. Random replay remained the strongest baseline in the current setup. Risk-aware replay became more competitive when combined with class balance or random diversity.

## Repository Structure

```text
notebooks/          Main experiment notebook
src/                Reusable code
results/figures/    Saved plots
results/tables/     Saved CSV tables