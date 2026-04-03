#!/usr/bin/env node

/**
 * TrustMRR MCP Server
 * Wraps the TrustMRR API (https://trustmrr.com/api/v1) and exposes two tools:
 *   - list_startups  : paginated startup directory with filters and sorting
 *   - get_startup    : full detail for a single startup by slug
 *
 * Transport: stdio (required for MCPB / Claude Desktop)
 * Auth: Bearer token injected via TRUSTMRR_API_KEY env var (set by manifest user_config)
 */

"use strict";

// ---------------------------------------------------------------------------
// Process stability — must come before anything else
// ---------------------------------------------------------------------------

// Prevent stdin.pause() (called by StdioServerTransport.close()) from
// draining the Node event loop and killing the process mid-session.
process.stdin.resume();

// In Node 15+, unhandled promise rejections terminate the process.
// Catch them here so a single SDK-internal error doesn't bring down the server.
process.on("unhandledRejection", (reason) => {
  process.stderr.write(
    `[trustmrr-mcp] Unhandled rejection: ${reason instanceof Error ? reason.stack : reason}\n`
  );
});

process.on("uncaughtException", (err) => {
  process.stderr.write(`[trustmrr-mcp] Uncaught exception: ${err.stack || err}\n`);
});

const { Server } = require("@modelcontextprotocol/sdk/server/index.js");
const { StdioServerTransport } = require("@modelcontextprotocol/sdk/server/stdio.js");
const {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  McpError,
  ErrorCode,
} = require("@modelcontextprotocol/sdk/types.js");

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const BASE_URL = "https://trustmrr.com/api/v1";
const TIMEOUT_MS = 15_000;
const SERVER_NAME = "trustmrr-mcp";
const SERVER_VERSION = "1.0.0";

const API_KEY = process.env.TRUSTMRR_API_KEY;

// Warn to stderr (not stdout — that breaks stdio transport) if key is missing.
// The server still starts; each tool call will return a clear error.
if (!API_KEY) {
  process.stderr.write(
    "[trustmrr-mcp] WARNING: TRUSTMRR_API_KEY is not set. " +
    "Set it via the Claude Desktop extension settings.\n"
  );
}

// ---------------------------------------------------------------------------
// HTTP helper
// ---------------------------------------------------------------------------

/**
 * Lightweight fetch wrapper with timeout and structured error handling.
 * Returns parsed JSON on success, throws McpError on failure.
 */
async function apiFetch(path, params = {}) {
  if (!API_KEY) {
    throw new McpError(
      ErrorCode.InvalidRequest,
      "TRUSTMRR_API_KEY is not configured. " +
      "Open Claude Desktop settings for the TrustMRR extension and enter your API key."
    );
  }

  // Build URL
  const url = new URL(`${BASE_URL}${path}`);
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== "") {
      url.searchParams.set(k, String(v));
    }
  }

  // AbortController for timeout
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);

  let response;
  try {
    response = await fetch(url.toString(), {
      method: "GET",
      headers: {
        Authorization: `Bearer ${API_KEY}`,
        "Content-Type": "application/json",
        "User-Agent": `${SERVER_NAME}/${SERVER_VERSION}`,
      },
      signal: controller.signal,
    });
  } catch (err) {
    clearTimeout(timer);
    if (err.name === "AbortError") {
      throw new McpError(
        ErrorCode.InternalError,
        `TrustMRR API request timed out after ${TIMEOUT_MS / 1000}s.`
      );
    }
    throw new McpError(
      ErrorCode.InternalError,
      `Network error reaching TrustMRR API: ${err.message}`
    );
  }
  clearTimeout(timer);

  // Rate limit header logging (helpful for debugging)
  const remaining = response.headers.get("X-RateLimit-Remaining");
  const reset = response.headers.get("X-RateLimit-Reset");
  if (remaining !== null) {
    process.stderr.write(
      `[trustmrr-mcp] Rate limit: ${remaining} requests remaining` +
      (reset ? `, resets at ${new Date(Number(reset) * 1000).toISOString()}` : "") +
      "\n"
    );
  }

  // Parse body
  let body;
  try {
    body = await response.json();
  } catch {
    throw new McpError(
      ErrorCode.InternalError,
      `TrustMRR API returned non-JSON response (status ${response.status}).`
    );
  }

  // Map HTTP errors to MCP errors with actionable messages
  if (!response.ok) {
    const msg = body?.error || "Unknown error";
    switch (response.status) {
      case 400:
        throw new McpError(ErrorCode.InvalidParams, `Bad request: ${msg}`);
      case 401:
        throw new McpError(
          ErrorCode.InvalidRequest,
          "TrustMRR API key is missing or invalid. " +
          "Regenerate your key at https://trustmrr.com/developer."
        );
      case 404:
        throw new McpError(ErrorCode.InvalidParams, `Not found: ${msg}`);
      case 429:
        throw new McpError(
          ErrorCode.InternalError,
          "TrustMRR rate limit hit (20 req/min). Wait a moment and try again."
        );
      case 500:
        throw new McpError(
          ErrorCode.InternalError,
          "TrustMRR server error. Try again shortly."
        );
      default:
        throw new McpError(
          ErrorCode.InternalError,
          `TrustMRR API error ${response.status}: ${msg}`
        );
    }
  }

  return body;
}

// ---------------------------------------------------------------------------
// Value formatters (cents -> USD strings for readability in LLM output)
// ---------------------------------------------------------------------------

function centsToUSD(cents) {
  if (cents === null || cents === undefined) return null;
  return `$${(cents / 100).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

/**
 * Enrich a startup object:
 * - Add human-readable USD fields alongside the raw cent values
 * - Leave all original fields intact so downstream callers can still use raw data
 */
function enrichStartup(s) {
  if (!s) return s;
  return {
    ...s,
    revenue: s.revenue
      ? {
          ...s.revenue,
          last30Days_usd: centsToUSD(s.revenue.last30Days),
          mrr_usd: centsToUSD(s.revenue.mrr),
          total_usd: centsToUSD(s.revenue.total),
        }
      : s.revenue,
    askingPrice_usd: centsToUSD(s.askingPrice),
  };
}

// ---------------------------------------------------------------------------
// Tool definitions (schema shown to the LLM via ListTools)
// ---------------------------------------------------------------------------

const TOOLS = [
  {
    name: "list_startups",
    description:
      "Browse and filter the TrustMRR startup directory. Returns paginated results with verified revenue, MRR, growth, asking price, tech stack thumbnail, and more. " +
      "All monetary values include both raw USD cents (field) and human-readable USD strings (field_usd). " +
      "Use onSale=true + sort=best-deal to find acquisition targets. " +
      "Use category=crypto-web3 or category=health-fitness for niche filters. " +
      "Paginate with page + limit (max 50 per page).",
    inputSchema: {
      type: "object",
      properties: {
        page: {
          type: "integer",
          minimum: 1,
          default: 1,
          description: "Page number, starts at 1.",
        },
        limit: {
          type: "integer",
          minimum: 1,
          maximum: 50,
          default: 10,
          description: "Results per page (1-50).",
        },
        sort: {
          type: "string",
          enum: [
            "revenue-desc",
            "revenue-asc",
            "price-desc",
            "price-asc",
            "multiple-asc",
            "multiple-desc",
            "growth-desc",
            "growth-asc",
            "listed-desc",
            "listed-asc",
            "best-deal",
          ],
          default: "revenue-desc",
          description:
            "Sort order. Default is revenue-desc. When onSale=true the default shifts to best-deal.",
        },
        onSale: {
          type: "string",
          enum: ["true", "false"],
          description:
            'Filter by sale status. "true" = for sale only, "false" = not for sale, omit for all.',
        },
        category: {
          type: "string",
          enum: [
            "ai", "saas", "developer-tools", "fintech", "marketing",
            "ecommerce", "productivity", "design-tools", "no-code", "analytics",
            "crypto-web3", "education", "health-fitness", "social-media",
            "content-creation", "sales", "customer-support", "recruiting",
            "real-estate", "travel", "legal", "security", "iot-hardware",
            "green-tech", "entertainment", "games", "community",
            "news-magazines", "utilities", "marketplace", "mobile-apps",
          ],
          description: "Filter by startup category.",
        },
        xHandle: {
          type: "string",
          description:
            'Filter by founder X (Twitter) handle (without @). Example: "marc_louvion".',
        },
        minRevenue: {
          type: "number",
          description:
            "Minimum last-30-days revenue in USD cents. Example: 100000 = $1,000.",
        },
        maxRevenue: {
          type: "number",
          description:
            "Maximum last-30-days revenue in USD cents. Example: 500000 = $5,000.",
        },
        minMrr: {
          type: "number",
          description:
            "Minimum MRR in USD cents. Example: 50000 = $500/mo.",
        },
        maxMrr: {
          type: "number",
          description:
            "Maximum MRR in USD cents. Example: 1000000 = $10,000/mo.",
        },
        minGrowth: {
          type: "number",
          description:
            "Minimum 30-day revenue growth as a decimal. Example: 0.1 = 10% growth.",
        },
        maxGrowth: {
          type: "number",
          description:
            "Maximum 30-day revenue growth as a decimal. Example: 0.5 = 50% growth.",
        },
        minPrice: {
          type: "number",
          description:
            "Minimum asking price in USD cents. Example: 1000000 = $10,000.",
        },
        maxPrice: {
          type: "number",
          description:
            "Maximum asking price in USD cents. Example: 10000000 = $100,000.",
        },
      },
      additionalProperties: false,
    },
  },
  {
    name: "get_startup",
    description:
      "Get full details for a single startup by its slug (URL-friendly identifier). " +
      "Returns everything from list_startups plus: full untruncated description, tech stack array, cofounders array, " +
      "X follower count, and isMerchantOfRecord. Use this after list_startups to drill into a specific startup.",
    inputSchema: {
      type: "object",
      required: ["slug"],
      properties: {
        slug: {
          type: "string",
          description:
            'URL-friendly startup identifier from the list_startups slug field. Example: "shipfast".',
          minLength: 1,
        },
      },
      additionalProperties: false,
    },
  },
];

// ---------------------------------------------------------------------------
// Tool handlers
// ---------------------------------------------------------------------------

async function handleListStartups(args) {
  const {
    page, limit, sort, onSale, category, xHandle,
    minRevenue, maxRevenue, minMrr, maxMrr,
    minGrowth, maxGrowth, minPrice, maxPrice,
  } = args || {};

  const params = {
    ...(page !== undefined && { page }),
    ...(limit !== undefined && { limit }),
    ...(sort && { sort }),
    ...(onSale && { onSale }),
    ...(category && { category }),
    ...(xHandle && { xHandle }),
    ...(minRevenue !== undefined && { minRevenue }),
    ...(maxRevenue !== undefined && { maxRevenue }),
    ...(minMrr !== undefined && { minMrr }),
    ...(maxMrr !== undefined && { maxMrr }),
    ...(minGrowth !== undefined && { minGrowth }),
    ...(maxGrowth !== undefined && { maxGrowth }),
    ...(minPrice !== undefined && { minPrice }),
    ...(maxPrice !== undefined && { maxPrice }),
  };

  const result = await apiFetch("/startups", params);

  const enrichedData = (result.data || []).map(enrichStartup);

  return {
    startups: enrichedData,
    pagination: result.meta || null,
    _note:
      "All revenue/price fields ending in _usd are human-readable strings. " +
      "Raw cent fields (last30Days, mrr, total, askingPrice) are also included for programmatic use.",
  };
}

async function handleGetStartup(args) {
  const { slug } = args || {};

  if (!slug || typeof slug !== "string" || slug.trim() === "") {
    throw new McpError(ErrorCode.InvalidParams, "slug is required and must be a non-empty string.");
  }

  // Sanitize slug — only allow alphanumeric and hyphens to prevent path traversal
  const sanitized = slug.trim().toLowerCase();
  if (!/^[a-z0-9-]+$/.test(sanitized)) {
    throw new McpError(
      ErrorCode.InvalidParams,
      `Invalid slug format: "${slug}". Slugs contain only lowercase letters, numbers, and hyphens.`
    );
  }

  const result = await apiFetch(`/startups/${encodeURIComponent(sanitized)}`);
  const enriched = enrichStartup(result.data);

  return {
    startup: enriched,
    _note:
      "All revenue/price fields ending in _usd are human-readable strings. " +
      "techStack and cofounders are only available on this detail endpoint.",
  };
}

// ---------------------------------------------------------------------------
// MCP Server bootstrap
// ---------------------------------------------------------------------------

async function main() {
  const server = new Server(
    { name: SERVER_NAME, version: SERVER_VERSION },
    {
      capabilities: {
        tools: {},
      },
    }
  );

  // List tools
  server.setRequestHandler(ListToolsRequestSchema, async () => ({
    tools: TOOLS,
  }));

  // Call tool
  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: args } = request.params;

    process.stderr.write(`[trustmrr-mcp] Tool call: ${name} ${JSON.stringify(args)}\n`);

    try {
      let result;
      switch (name) {
        case "list_startups":
          result = await handleListStartups(args);
          break;
        case "get_startup":
          result = await handleGetStartup(args);
          break;
        default:
          throw new McpError(ErrorCode.MethodNotFound, `Unknown tool: "${name}"`);
      }

      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(result, null, 2),
          },
        ],
      };
    } catch (err) {
      // Re-throw McpErrors directly — the SDK will serialize them correctly
      if (err instanceof McpError) throw err;

      // Unexpected errors
      process.stderr.write(`[trustmrr-mcp] Unexpected error in ${name}: ${err.stack || err}\n`);
      throw new McpError(
        ErrorCode.InternalError,
        `Unexpected error: ${err.message || String(err)}`
      );
    }
  });

  // Connect stdio transport
  const transport = new StdioServerTransport();

  process.stderr.write(`[trustmrr-mcp] Starting server v${SERVER_VERSION} via stdio...\n`);

  await server.connect(transport);

  process.stderr.write("[trustmrr-mcp] Server connected and ready.\n");

  // Keep the process alive for the full session lifetime.
  //
  // Why this is necessary:
  //   StdioServerTransport.close() calls stdin.pause(), which removes stdin
  //   from Node's event loop. If nothing else is keeping the loop alive, Node
  //   exits with code 0 — no crash, no error log, just a silent disconnect.
  //   Awaiting this promise ensures the process only exits after a deliberate
  //   transport close or a OS signal.
  await new Promise((resolve) => {
    const shutdown = () => {
      process.stderr.write("[trustmrr-mcp] Shutting down.\n");
      resolve();
    };

    // Resolve when the transport closes (client disconnected or server closed)
    const originalOnClose = transport.onclose;
    transport.onclose = () => {
      originalOnClose?.();
      shutdown();
    };

    // Also resolve on OS signals so the process exits cleanly
    process.once("SIGTERM", shutdown);
    process.once("SIGINT", shutdown);
  });
}

main().catch((err) => {
  process.stderr.write(`[trustmrr-mcp] Fatal error: ${err.stack || err}\n`);
  process.exit(1);
});
