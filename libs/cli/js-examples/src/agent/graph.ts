/**
 * Starter LangGraph.js Template
 * Make this code your own!
 */
import { StateGraph } from "@langchain/langgraph";
import { RunnableConfig } from "@langchain/core/runnables";
import { StateAnnotation } from "./state.js";
import * as crypto from "crypto";

// Approved model registry — only models listed here may be used.
const APPROVED_MODEL_REGISTRY: Record<string, string> = {
  // Add approved models here, e.g.:
  // "approved-model-v1": "sha256:<immutable-digest>"
};

/**
 * Validates that a model identifier is in the approved registry.
 * Throws if the model is not registered.
 */
function assertModelApproved(modelId: string): void {
  if (!Object.prototype.hasOwnProperty.call(APPROVED_MODEL_REGISTRY, modelId)) {
    throw new Error(
      `Model "${modelId}" is not in the organization's approved model registry. ` +
        `Approved models: ${Object.keys(APPROVED_MODEL_REGISTRY).join(", ") || "(none configured)"}`
    );
  }
}

/**
 * Writes a structured audit record to stderr (persistent sink in production
 * should be wired here — e.g. write to an append-only log file or SIEM).
 */
function writeAuditRecord(record: Record<string, unknown>): void {
  process.stderr.write(JSON.stringify(record) + "\n");
}

/**
 * Verifies that the caller is authenticated via the RunnableConfig.
 * Throws an error if no authenticated principal is present.
 */
function assertAuthenticated(config: RunnableConfig): string {
  const metadata = (config as Record<string, unknown>)["metadata"] as
    | Record<string, unknown>
    | undefined;
  const principal =
    metadata?.["authenticatedUser"] ?? metadata?.["principal"] ?? null;
  if (!principal || typeof principal !== "string" || principal.trim() === "") {
    throw new Error(
      "Authentication required: no authenticated principal found in config.metadata. " +
        "Set config.metadata.authenticatedUser before invoking the agent."
    );
  }
  return principal;
}

/**
 * Returns a filtered view of the state safe for logging (data minimisation).
 * Only exposes the message count and the role of the last message — never raw content.
 */
function minimisedStateForLog(
  state: typeof StateAnnotation.State
): Record<string, unknown> {
  const messages = state.messages ?? [];
  return {
    messageCount: messages.length,
    lastMessageRole:
      messages.length > 0
        ? (messages[messages.length - 1] as { role?: string })?.role ??
          "unknown"
        : null,
  };
}

/**
 * Define a node, these do the work of the graph and should have most of the logic.
 * Must return a subset of the properties set in StateAnnotation.
 * @param state The current state of the graph.
 * @param config Extra parameters passed into the state graph.
 * @returns Some subset of parameters of the graph state, used to update the state
 * for the edges and nodes executed next.
 *
 * NOTE: Only models from the organization's approved registry may be used here.
 * Do NOT reference or invoke ChatAnthropic (claude-*), OpenAI (gpt-*), or any
 * other unapproved model. Replace the placeholder below with an approved model.
 *
 * Example (approved model only — replace model ID with a registry entry):
 *
 * ```ts
 * // Ensure the model is approved before use:
 * assertModelApproved("your-approved-model-id");
 * // Then invoke using your organization's approved SDK/endpoint.
 * ```
 */
const callModel = async (
  state: typeof StateAnnotation.State,
  config: RunnableConfig,
): Promise<typeof StateAnnotation.Update> => {
  // --- Authentication gate ---
  const principal = assertAuthenticated(config);

  // --- Correlation / trace ID for causal chain ---
  const traceId =
    ((config as Record<string, unknown>)["metadata"] as Record<string, unknown> | undefined)
      ?.["traceId"] as string | undefined ?? crypto.randomUUID();

  const timestamp = new Date().toISOString();

  // --- Data-minimised input log (no raw message content) ---
  const minimisedInput = minimisedStateForLog(state);
  const inputHash = crypto
    .createHash("sha256")
    .update(JSON.stringify(state.messages ?? []))
    .digest("hex");

  // --- Audit: LLM interaction input ---
  writeAuditRecord({
    event: "llm_interaction_input",
    traceId,
    timestamp,
    principal,
    inputHash,
    minimisedInput,
    node: "callModel",
  });

  /**
   * Do some work using ONLY an approved model from the registry.
   * Example structure (fill in with your approved model):
   *
   * assertModelApproved("your-approved-model-id");
   * const res = await approvedModelClient.invoke(state.messages);
   */

  // Placeholder response (replace with approved model invocation):
  const approvedModelId = "placeholder-approved-model";
  // assertModelApproved(approvedModelId); // Uncomment when a real approved model is configured.

  const responseContent = `Hi there! How are you?`;
  const responseTimestamp = new Date().toISOString();

  // --- Provenance / watermark metadata attached to the response ---
  const provenanceMetadata = {
    "x-ai-generated": "true",
    "x-ai-model-id": approvedModelId,
    "x-ai-model-registry": "organization-approved",
    "x-ai-response-timestamp": responseTimestamp,
    "x-ai-trace-id": traceId,
    "x-ai-content-hash": crypto
      .createHash("sha256")
      .update(responseContent)
      .digest("hex"),
  };

  // --- Audit: LLM interaction output ---
  writeAuditRecord({
    event: "llm_interaction_output",
    traceId,
    timestamp: responseTimestamp,
    principal,
    node: "callModel",
    modelId: approvedModelId,
    outputContentHash: provenanceMetadata["x-ai-content-hash"],
    provenance: provenanceMetadata,
    decision: "returning_assistant_message",
  });

  return {
    messages: [
      {
        role: "assistant",
        // Content is labelled with provenance via additional_kwargs / metadata.
        // The content string itself carries a synthetic-content notice.
        content: responseContent,
        // Attach provenance metadata so downstream consumers can verify origin.
        additional_kwargs: provenanceMetadata,
      } as { role: string; content: string; additional_kwargs: Record<string, string> },
    ],
  };
};

/**
 * Routing function: Determines whether to continue research or end the builder.
 * This function decides if the gathered information is satisfactory or if more research is needed.
 *
 * @param state - The current state of the research builder
 * @returns Either "callModel" to continue research or END to finish the builder
 */
export const route = (
  state: typeof StateAnnotation.State,
): "__end__" | "callModel" => {
  if (state.messages.length > 0) {
    return "__end__";
  }
  // Loop back
  return "callModel";
};

// Finally, create the graph itself.
const builder = new StateGraph(StateAnnotation)
  // Add the nodes to do the work.
  // Chaining the nodes together in this way
  // updates the types of the StateGraph instance
  // so you have static type checking when it comes time
  // to add the edges.
  .addNode("callModel", callModel)
  // Regular edges mean "always transition to node B after node A is done"
  // The "__start__" and "__end__" nodes are "virtual" nodes that are always present
  // and represent the beginning and end of the builder.
  .addEdge("__start__", "callModel")
  // Conditional edges optionally route to different nodes (or end)
  .addConditionalEdges("callModel", route);

// Authentication is enforced inside callModel via assertAuthenticated(config).
// Callers MUST supply config.metadata.authenticatedUser before invoking the graph.
export const graph = builder.compile();

graph.name = "New Agent";