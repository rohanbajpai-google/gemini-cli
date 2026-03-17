/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
/* eslint-disable @typescript-eslint/no-unsafe-type-assertion */
/* eslint-disable @typescript-eslint/no-unsafe-assignment */
/* eslint-disable @typescript-eslint/await-thenable */

import { AnthropicVertex } from '@anthropic-ai/vertex-sdk';
import type {
  MessageParam,
  Tool as AnthropicTool,
  ToolResultBlockParam,
  ToolUseBlockParam,
  TextBlockParam,
  ImageBlockParam,
  MessageCreateParamsNonStreaming,
  MessageCreateParamsStreaming,
  Message,
} from '@anthropic-ai/sdk/resources/messages';

import type {
  Content,
  CountTokensParameters,
  CountTokensResponse,
  EmbedContentParameters,
  EmbedContentResponse,
  FunctionCall,
  GenerateContentParameters,
  GenerateContentResponse,
  Part,
  Tool,
  ContentListUnion,
  ToolListUnion,
} from '@google/genai';
import type { ContentGenerator } from './contentGenerator.js';
import type { LlmRole } from '../telemetry/llmRole.js';

/** Anthropic image media types supported by the Messages API. */
type AnthropicImageMediaType =
  | 'image/jpeg'
  | 'image/png'
  | 'image/gif'
  | 'image/webp';

const DEFAULT_ANTHROPIC_MODEL = 'claude-3-5-sonnet-v2@20241022';
const DEFAULT_MAX_TOKENS = 8192;
/** Rough chars-per-token estimate for the token counting fallback. */
const CHARS_PER_TOKEN_ESTIMATE = 4;

/**
 * Extended FunctionCall that carries the Anthropic tool-use `id` so it can be
 * round-tripped back in a `tool_result` block.
 */
interface FunctionCallWithId extends FunctionCall {
  id: string;
}

/**
 * Content generator that adapts the Gemini `GenerateContentParameters` /
 * `GenerateContentResponse` interface to the Anthropic Messages API via the
 * Vertex AI SDK.
 */
export class AnthropicContentGenerator implements ContentGenerator {
  private readonly client: AnthropicVertex;

  constructor(projectId: string, location: string) {
    this.client = new AnthropicVertex({
      projectId,
      region: location,
    });
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  async generateContent(
    request: GenerateContentParameters,
    _userPromptId: string,
    _role: LlmRole,
  ): Promise<GenerateContentResponse> {
    const params = this.buildRequestParams(request);
    const response = await this.client.messages.create(params);
    return this.mapAnthropicResponseToGemini(response);
  }

  async generateContentStream(
    request: GenerateContentParameters,
    _userPromptId: string,
    _role: LlmRole,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    const params: MessageCreateParamsStreaming = {
      ...this.buildRequestParams(request),
      stream: true,
    };
    const stream = await this.client.messages.stream(params);
    return this.streamChunks(stream);
  }

  async countTokens(
    request: CountTokensParameters,
  ): Promise<CountTokensResponse> {
    const textStr = this.extractTextForCounting(request);
    const count = Math.ceil(textStr.length / CHARS_PER_TOKEN_ESTIMATE);
    return { totalTokens: count };
  }

  async embedContent(
    _request: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    throw new Error(
      'Embeddings are not supported via the Anthropic Vertex SDK.',
    );
  }

  // ---------------------------------------------------------------------------
  // Request building helpers
  // ---------------------------------------------------------------------------

  /**
   * Builds the shared Anthropic request parameters from a Gemini-style request.
   */
  private buildRequestParams(
    request: GenerateContentParameters,
  ): MessageCreateParamsNonStreaming {
    const messages = this.mapContentsToMessages(request.contents || []);
    const tools = this.mapToolDeclarations(request.config?.tools);
    const system = this.extractSystemPrompt(request);
    const config = request.config;

    const params: MessageCreateParamsNonStreaming = {
      model: request.model || DEFAULT_ANTHROPIC_MODEL,
      max_tokens: config?.maxOutputTokens || DEFAULT_MAX_TOKENS,
      messages,
    };

    if (tools) params.tools = tools;
    if (system) params.system = system;
    // Anthropic does not allow both temperature and top_p simultaneously.
    // Prefer temperature when both are specified.
    if (config?.temperature !== undefined) {
      params.temperature = config.temperature;
    } else if (config?.topP !== undefined) {
      params.top_p = config.topP;
    }
    if (config?.topK !== undefined) params.top_k = config.topK;

    return params;
  }

  // ---------------------------------------------------------------------------
  // Gemini → Anthropic mapping
  // ---------------------------------------------------------------------------

  /**
   * Converts Gemini `Part[]` into Anthropic content blocks.
   */
  private mapPartsToBlocks(
    parts: Part[],
  ): Array<
    TextBlockParam | ImageBlockParam | ToolUseBlockParam | ToolResultBlockParam
  > {
    const blocks: Array<
      | TextBlockParam
      | ImageBlockParam
      | ToolUseBlockParam
      | ToolResultBlockParam
    > = [];

    for (const part of parts) {
      if (part.text) {
        blocks.push({ type: 'text', text: part.text });
      } else if (part.inlineData) {
        blocks.push({
          type: 'image',
          source: {
            type: 'base64',
            media_type: (part.inlineData.mimeType ||
              'image/jpeg') as AnthropicImageMediaType,
            data: part.inlineData.data || '',
          },
        });
      } else if (part.functionCall) {
        const fc = part.functionCall as FunctionCallWithId;
        blocks.push({
          type: 'tool_use',
          id: fc.id || fc.name || '',
          name: fc.name || '',
          input: (fc.args as Record<string, unknown>) || {},
        });
      } else if (part.functionResponse) {
        const fr = part.functionResponse as {
          id?: string;
        } & typeof part.functionResponse;
        blocks.push({
          type: 'tool_result',
          tool_use_id: fr.id || fr.name || '',
          content:
            typeof fr.response === 'string'
              ? fr.response
              : JSON.stringify(fr.response),
        });
      } else if (part.fileData) {
        blocks.push({
          type: 'text',
          text: `[File data omitted: ${part.fileData.fileUri}]`,
        });
      }
    }

    return blocks;
  }

  /**
   * Converts a Gemini `ContentListUnion` into Anthropic `MessageParam[]`.
   *
   * The Anthropic Messages API enforces strict constraints:
   *   1. Messages must alternate between `user` and `assistant` roles.
   *   2. Every `tool_use` block in an `assistant` message must have a
   *      corresponding `tool_result` block in the immediately following
   *      `user` message.
   *
   * Gemini history may violate these constraints due to curation (dropping
   * invalid model turns), context injection (IDE context, directory context),
   * or error/abort scenarios. This method merges consecutive same-role
   * messages and injects synthetic `tool_result` blocks when tool_use IDs
   * are left unmatched.
   */
  private mapContentsToMessages(contents: ContentListUnion): MessageParam[] {
    const contentsArray: Content[] = Array.isArray(contents)
      ? (contents as Content[])
      : [contents as Content];

    // Step 1: Map to Anthropic messages and merge consecutive same-role turns.
    const merged: MessageParam[] = [];
    for (const content of contentsArray) {
      const role =
        content.role === 'model' ? ('assistant' as const) : ('user' as const);
      const blocks = this.mapPartsToBlocks(content.parts || []);

      const last = merged[merged.length - 1];
      if (last && last.role === role && Array.isArray(last.content)) {
        (
          last.content as Array<
            | TextBlockParam
            | ImageBlockParam
            | ToolUseBlockParam
            | ToolResultBlockParam
          >
        ).push(...blocks);
      } else {
        merged.push({ role, content: blocks });
      }
    }

    // Step 2: Ensure every tool_use has a matching tool_result immediately
    // after. Walk the merged messages and inject synthetic tool_result blocks
    // where they are missing.
    const result: MessageParam[] = [];
    for (let i = 0; i < merged.length; i++) {
      const msg = merged[i];
      result.push(msg);

      if (msg.role !== 'assistant' || !Array.isArray(msg.content)) continue;

      // Collect tool_use IDs from this assistant message.
      const toolUseIds: string[] = [];
      for (const block of msg.content) {
        if ((block as ToolUseBlockParam).type === 'tool_use') {
          toolUseIds.push((block as ToolUseBlockParam).id);
        }
      }
      if (toolUseIds.length === 0) continue;

      // Check the next message for matching tool_result blocks.
      const next = merged[i + 1];
      const matchedIds = new Set<string>();
      if (next && next.role === 'user' && Array.isArray(next.content)) {
        for (const block of next.content) {
          if ((block as ToolResultBlockParam).type === 'tool_result') {
            matchedIds.add((block as ToolResultBlockParam).tool_use_id);
          }
        }
      }

      // Inject synthetic tool_result blocks for any unmatched tool_use IDs.
      const unmatchedIds = toolUseIds.filter((id) => !matchedIds.has(id));
      if (unmatchedIds.length > 0) {
        const syntheticResults: ToolResultBlockParam[] = unmatchedIds.map(
          (id) => ({
            type: 'tool_result' as const,
            tool_use_id: id,
            content: 'No result available.',
          }),
        );

        if (next && next.role === 'user' && Array.isArray(next.content)) {
          // Prepend synthetic results to the existing user message.
          (
            next.content as Array<
              | TextBlockParam
              | ImageBlockParam
              | ToolUseBlockParam
              | ToolResultBlockParam
            >
          ).unshift(...syntheticResults);
        } else {
          // No user message follows; insert a new one.
          const syntheticUserMsg: MessageParam = {
            role: 'user',
            content: syntheticResults,
          };
          // Insert it right after the current assistant message (before
          // whatever comes next). We already pushed `msg` into `result`, so
          // just push the synthetic user message. The next iteration will
          // handle `merged[i+1]` normally.
          result.push(syntheticUserMsg);
        }
      }
    }

    return result;
  }

  /**
   * Converts Gemini tool declarations into Anthropic `Tool[]`.
   */
  private mapToolDeclarations(
    tools?: ToolListUnion,
  ): AnthropicTool[] | undefined {
    if (!tools || !Array.isArray(tools) || tools.length === 0) return undefined;

    const mapped: AnthropicTool[] = [];

    for (const tool of tools as Tool[]) {
      if (!tool.functionDeclarations) continue;
      for (const func of tool.functionDeclarations) {
        mapped.push({
          name: func.name || '',
          description: func.description,
          input_schema:
            (func.parameters as unknown as AnthropicTool.InputSchema) || {
              type: 'object' as const,
              properties: {},
            },
        });
      }
    }

    return mapped.length > 0 ? mapped : undefined;
  }

  /**
   * Extracts the system prompt string from the Gemini request config.
   */
  private extractSystemPrompt(
    request: GenerateContentParameters,
  ): string | undefined {
    const sysInst = request.config?.systemInstruction;
    if (!sysInst) return undefined;
    if (typeof sysInst === 'string') return sysInst;

    let parts: Part[];
    if (Array.isArray(sysInst)) {
      parts = sysInst as Part[];
    } else if (
      typeof sysInst === 'object' &&
      'parts' in (sysInst as Record<string, unknown>)
    ) {
      parts =
        ((sysInst as unknown as Record<string, unknown>)['parts'] as Part[]) ||
        [];
    } else {
      parts = [sysInst as unknown as Part];
    }

    return parts.map((p) => p.text || '').join('\n');
  }

  // ---------------------------------------------------------------------------
  // Anthropic → Gemini mapping
  // ---------------------------------------------------------------------------

  /**
   * Maps an Anthropic `Message` response to a Gemini `GenerateContentResponse`.
   */
  private mapAnthropicResponseToGemini(
    response: Message,
  ): GenerateContentResponse {
    const parts: Part[] = response.content.map((block) => {
      if (block.type === 'text') {
        return { text: block.text };
      }
      // block.type === 'tool_use'
      return {
        functionCall: {
          id: block.id,
          name: block.name,
          args: block.input as Record<string, unknown>,
        } as FunctionCall,
      };
    });

    return {
      candidates: [
        {
          content: { role: 'model', parts },
        },
      ],
      usageMetadata: {
        promptTokenCount: response.usage.input_tokens,
        candidatesTokenCount: response.usage.output_tokens,
        totalTokenCount:
          response.usage.input_tokens + response.usage.output_tokens,
      },
      get text() {
        return parts
          .filter((p) => p.text)
          .map((p) => p.text)
          .join('\n');
      },
      get functionCalls() {
        const fcs = parts
          .filter((p) => p.functionCall)
          .map((p) => p.functionCall);
        return fcs.length > 0 ? fcs : undefined;
      },
    } as unknown as GenerateContentResponse;
  }

  // ---------------------------------------------------------------------------
  // Streaming
  // ---------------------------------------------------------------------------

  /**
   * Processes an Anthropic message stream and yields Gemini-shaped response
   * chunks.
   */
  private async *streamChunks(
    stream: AsyncIterable<{ type: string; [key: string]: unknown }>,
  ): AsyncGenerator<GenerateContentResponse> {
    let currentFunctionId: string | undefined;
    let currentFunctionName: string | undefined;
    let currentFunctionArgsStr = '';

    for await (const chunk of stream) {
      switch (chunk.type) {
        case 'content_block_start': {
          const block = chunk['content_block'] as {
            type: string;
            id?: string;
            name?: string;
          };
          if (block.type === 'tool_use') {
            currentFunctionId = block.id;
            currentFunctionName = block.name;
            currentFunctionArgsStr = '';
          }
          break;
        }

        case 'content_block_delta': {
          const delta = chunk['delta'] as {
            type: string;
            text?: string;
            partial_json?: string;
          };
          if (delta.type === 'text_delta') {
            const deltaText = delta.text || '';
            yield {
              candidates: [
                {
                  content: {
                    role: 'model',
                    parts: [{ text: deltaText }],
                  },
                },
              ],
              get text() {
                return deltaText;
              },
              get functionCalls() {
                return undefined;
              },
            } as unknown as GenerateContentResponse;
          } else if (delta.type === 'input_json_delta') {
            currentFunctionArgsStr += delta.partial_json || '';
          }
          break;
        }

        case 'content_block_stop': {
          if (currentFunctionName) {
            let args: Record<string, unknown> = {};
            try {
              if (currentFunctionArgsStr) {
                args = JSON.parse(currentFunctionArgsStr) as Record<
                  string,
                  unknown
                >;
              }
            } catch {
              // Partial JSON may fail to parse; yield what we have.
            }

            const fc = {
              id: currentFunctionId,
              name: currentFunctionName,
              args,
            };

            yield {
              candidates: [
                {
                  content: {
                    role: 'model',
                    parts: [{ functionCall: fc }],
                  },
                },
              ],
              get text() {
                return '';
              },
              get functionCalls() {
                return [fc];
              },
            } as unknown as GenerateContentResponse;

            currentFunctionId = undefined;
            currentFunctionName = undefined;
            currentFunctionArgsStr = '';
          }
          break;
        }

        case 'message_delta': {
          const delta = chunk['delta'] as { stop_reason?: string };
          if (delta.stop_reason) {
            const finishReason =
              delta.stop_reason === 'max_tokens' ? 'MAX_TOKENS' : 'STOP';
            yield {
              candidates: [
                {
                  content: { role: 'model', parts: [] },
                  finishReason,
                },
              ],
            } as unknown as GenerateContentResponse;
          }
          break;
        }

        default:
          break;
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Token counting fallback
  // ---------------------------------------------------------------------------

  /**
   * Extracts raw text from a `CountTokensParameters` request for the
   * approximate token-count heuristic.
   */
  private extractTextForCounting(request: CountTokensParameters): string {
    if (Array.isArray(request.contents)) {
      return request.contents
        .map((c) => {
          const content = c as Content;
          return content.parts
            ? content.parts.map((p) => p.text || '').join(' ')
            : '';
        })
        .join('\n');
    }

    return typeof request.contents === 'string'
      ? request.contents
      : JSON.stringify(request.contents);
  }
}
