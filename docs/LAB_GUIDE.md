# Step-by-step lab guide

Read the [illustrated workbook](NovaMart_AgentCore_Learning_Workbook.pdf) alongside this checklist. Run commands from the repository root.

| Step | Tool | Action | Checkpoint |
|---|---|---|---|
| 1 | AWS CLI / CloudShell | Authenticate; run `aws sts get-caller-identity` | Correct account and region |
| 2 | Python / pip | Create a virtual environment; install requirements | Imports succeed |
| 3 | CloudFormation | Deploy `infrastructure/starter_stack.yaml` | `CREATE_COMPLETE` |
| 4 | Python / boto3 | Run `infrastructure/seed_data.py` | Orders, customers, policy documents exist |
| 5 | `config.py` | Read CloudFormation exports | Resource names match your stack |
| 6 | Bedrock Console | Create three self-managed KBs with matching S3 prefixes/indexes | Three KBs available |
| 7 | Bedrock Console | Sync all data sources | Successful sync and nonempty retrieval |
| 8 | Local editor | Populate `.env` KB IDs | `task5` can retrieve all three domains |
| 9 | Strands SDK | Inspect the five builders and three retriever agents | Correct tools, temperatures, model constants |
| 10 | DynamoDB | Inspect a workflow session | Worker fields and increasing version |
| 11 | AgentCore CLI / Node / uv | Run the deploy command | Runtime READY; ARN recorded |
| 12 | Guardrails | Check published numbered version and policies | Arithmetic allowed, denied topics blocked |
| 13 | CloudWatch / X-Ray | Run live scenarios and inspect traces | Logs and connected service nodes |
| 14 | Python test runner | Run `python tests/test_agent.py all` | Capture your own test output |
| 15 | Private submission | Collect source, populated `.env`, score screenshot, X-Ray screenshot | All required files included |

## Agent tools versus developer tools

Developer tools install, configure, deploy, and inspect the application. Agent tools are Python functions decorated with `@tool` and invoked by Strands during inference.

- Inventory: `check_order_status`, `get_customer_tier`, `list_customer_orders`.
- Refund: `get_inventory_context`, `initiate_refund`.
- Policy: `search_all_policies`, backed by one KB tool in each retriever.
- Communication: `get_full_workflow_context`.
- Orchestrator: `initialize_session` and the four `route_to_*_agent` tools.

## Expected routing exercises

1. Return `ORD-27176` as `CUST-001`: Inventory → Refund → Communication.
2. Premium return policy: Policy → three parallel retrievers → Communication.
3. Five items at $29.99 with 10% off: Communication; expected final amount $134.96.
4. "Am I Premium?": Inventory → Communication.

Every request initializes a session first. Communication is the final worker. An order-status request must not initiate a refund simply because the routing graph includes the Refund worker.

## Debugging order

1. Capture the exact error and failing command.
2. Separate local execution, runtime deployment, runtime invocation, and rubric tests.
3. Verify identity, region, model availability, and `.env` values.
4. For runtime HTTP 500 errors, inspect the runtime stdout log group and traceback.
5. For missing RAG results, verify the KB prefix, ingestion job, index, and IDs.
6. For build issues, check disk space, `uv` on PATH, and CDK development dependencies (`npm ci --include=dev` inside the CDK directory).
7. Re-run the smallest relevant check before repeating the full deployment.
