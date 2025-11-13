---
applyTo: '**'
---
You are a helpful agent in creating a novel cybersecurity benchmark that challenges llm agents to find conduct root-cause analysis based a given fuzzer report and the codebase. Our system should have three layers: a evaluation server that orchestrates the evaluation, an agent environment that contains the llm and the scaffold, and a test environment containing the vulnerable codebase and the fuzzer report, all sit in their own docker containers. However, the agent environment and the test environment are connected through a docker compose. 

Follow the following contribution guidelines:
1) Remember that this is a research project, so prioritize clarity and modularity over optimization.
2) Ensure all code is well-documented, with clear comments explaining the purpose of functions and
3) Keep the repo tight and avoid unnecessary files. Use .dockerignore to exclude files not needed in the docker context.
4) Manage secerets carefully. Use environment variables and .env files, but do not commit sensitive information to the repo.

Additionally, always run "conda activate rcabench" before executing any scripts to ensure the correct environment is used.