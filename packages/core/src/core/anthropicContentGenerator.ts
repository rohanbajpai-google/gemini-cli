/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
/* eslint-disable @typescript-eslint/no-unsafe-type-assertion */
/* eslint-disable @typescript-eslint/no-unsafe-assignment */
/* eslint-disable object-shorthand */

/* eslint-disable default-case */
/* eslint-disable @typescript-eslint/await-thenable */
/* eslint-disable @typescript-eslint/no-unused-vars */

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
  GenerateContentParameters,
  GenerateContentResponse,
  Part,
  Tool,
  ContentListUnion,
  ToolListUnion,
} from '@google/genai';
import type { ContentGenerator } from './contentGenerator.js';
import type { LlmRole } from '../telemetry/llmRole.js';

export class AnthropicContentGenerator implements ContentGenerator {
  private client: AnthropicVertex;

  constructor(projectId: string, location: string, _accessToken?: string) {
    this.client = new AnthropicVertex({
      projectId,
      region: location,
    });
  }

  private mapGeminiPartsToAnthropicBlocks(
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
    for (const p of parts) {
      if (p.text) {
        blocks.push({
          type: 'text',
          text: p.text,
        });
      } else if (p.inlineData) {
        blocks.push({
          type: 'image',
          source: {
            type: 'base64',
            media_type: (p.inlineData.mimeType || 'image/jpeg') as
              | 'image/jpeg'
              | 'image/png'
              | 'image/gif'
              | 'image/webp',
            data: p.inlineData.data || '',
          },
        });
      } else if (p.functionCall) {
        blocks.push({
          type: 'tool_use',
          id:
            (p.functionCall as { id?: string }).id || p.functionCall.name || '',
          name: p.functionCall.name || '',
          input: (p.functionCall.args as Record<string, unknown>) || {},
        });
      } else if (p.functionResponse) {
        blocks.push({
          type: 'tool_result',
          tool_use_id:
            (p.functionResponse as { id?: string }).id ||
            p.functionResponse.name ||
            '',
          content:
            typeof p.functionResponse.response === 'string'
              ? p.functionResponse.response
              : JSON.stringify(p.functionResponse.response),
        });
      } else if (p.fileData) {
        blocks.push({
          type: 'text',
          text: `[File Data omitted: ${p.fileData.fileUri}]`,
        });
      }
    }
    return blocks;
  }

  private mapGeminiContentsToAnthropicMessages(
    contents: ContentListUnion,
  ): MessageParam[] {
    const messages: MessageParam[] = [];

    const contentsArray: Content[] = Array.isArray(contents)
      ? (contents as Content[])
      : [contents as Content];

    for (const content of contentsArray) {
      const role = content.role === 'model' ? 'assistant' : 'user';
      const blocks = this.mapGeminiPartsToAnthropicBlocks(content.parts || []);

      messages.push({
        role,
        content: blocks,
      });
    }

    return messages;
  }

  private mapGeminiToolsToAnthropicTools(
    tools?: ToolListUnion,
  ): AnthropicTool[] | undefined {
    if (!tools || !Array.isArray(tools) || tools.length === 0) return undefined;

    const anthropicTools: AnthropicTool[] = [];

    for (const tool of tools as Tool[]) {
      if (tool.functionDeclarations) {
        for (const func of tool.functionDeclarations) {
          anthropicTools.push({
            name: func.name || '',
            description: func.description,
            input_schema:
              (func.parameters as unknown as AnthropicTool.InputSchema) || {
                type: 'object',
                properties: {},
              },
          });
        }
      }
    }

    return anthropicTools.length > 0 ? anthropicTools : undefined;
  }

  private constructSystemPrompt(
    request: GenerateContentParameters,
  ): string | undefined {
    const sysInst = request.config?.systemInstruction;
    if (!sysInst) return undefined;

    if (typeof sysInst === 'string') {
      return sysInst;
    }

    let parts: Part[] = [];
    if (Array.isArray(sysInst)) {
      parts = sysInst as Part[];
    } else if ('parts' in (sysInst as unknown as Record<string, unknown>)) {
      parts =
        ((sysInst as unknown as Record<string, unknown>)['parts'] as Part[]) ||
        [];
    } else {
      parts = [sysInst as unknown as Part];
    }

    return parts.map((p) => p.text || '').join('\n');
  }

  async generateContent(
    request: GenerateContentParameters,
    _userPromptId: string,
    _role: LlmRole,
  ): Promise<GenerateContentResponse> {
    const messages = this.mapGeminiContentsToAnthropicMessages(
      request.contents || [],
    );
    const tools = this.mapGeminiToolsToAnthropicTools(request.config?.tools);
    const system = this.constructSystemPrompt(request);

    const anthropicParams: MessageCreateParamsNonStreaming = {
      model: request.model || 'claude-3-5-sonnet-v2@20241022',
      max_tokens: request.config?.maxOutputTokens || 8192,
      messages: messages,
    };

    if (tools) anthropicParams.tools = tools;
    if (system) anthropicParams.system = system;
    if (request.config?.temperature !== undefined) {
      anthropicParams.temperature = request.config.temperature;
    } else if (request.config?.topP !== undefined) {
      anthropicParams.top_p = request.config.topP;
    }
    if (request.config?.topK !== undefined) {
      anthropicParams.top_k = request.config.topK;
    }

    const response = await this.client.messages.create(anthropicParams);
    return this.mapAnthropicResponseToGemini(response);
  }

  async generateContentStream(
    request: GenerateContentParameters,
    _userPromptId: string,
    _role: LlmRole,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    const messages = this.mapGeminiContentsToAnthropicMessages(
      request.contents || [],
    );
    const tools = this.mapGeminiToolsToAnthropicTools(request.config?.tools);
    const system = this.constructSystemPrompt(request);

    const anthropicParams: MessageCreateParamsStreaming = {
      model: request.model || 'claude-3-5-sonnet-v2@20241022',
      max_tokens: request.config?.maxOutputTokens || 8192,
      messages: messages,
      stream: true,
    };

    if (tools) anthropicParams.tools = tools;
    if (system) anthropicParams.system = system;
    if (request.config?.temperature !== undefined) {
      anthropicParams.temperature = request.config.temperature;
    } else if (request.config?.topP !== undefined) {
      anthropicParams.top_p = request.config.topP;
    }
    if (request.config?.topK !== undefined) {
      anthropicParams.top_k = request.config.topK;
    }

    const stream = await this.client.messages.stream(anthropicParams);

    return (async function* () {
      let currentFunctionName: string | undefined;
      let currentFunctionArgsStr = '';

      for await (const chunk of stream) {
        switch (chunk.type) {
          case 'content_block_start':
            if (chunk.content_block.type === 'tool_use') {
              currentFunctionName = chunk.content_block.name;
              currentFunctionArgsStr = '';
            }
            break;
          case 'content_block_delta':
            if (chunk.delta.type === 'text_delta') {
              const deltaText = (chunk.delta as { text?: string }).text || '';
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
            } else if (chunk.delta.type === 'input_json_delta') {
              currentFunctionArgsStr +=
                (chunk.delta as { partial_json?: string }).partial_json || '';
            }
            break;
          case 'content_block_stop':
            if (currentFunctionName) {
              let args = {};
              try {
                if (currentFunctionArgsStr) {
                  args = JSON.parse(currentFunctionArgsStr);
                }
              } catch (e) {
                // Ignore partial json parse fail
              }
              yield {
                candidates: [
                  {
                    content: {
                      role: 'model',
                      parts: [
                        {
                          functionCall: {
                            name: currentFunctionName,
                            args: args,
                          },
                        },
                      ],
                    },
                  },
                ],
                get text() {
                  return '';
                },
                get functionCalls() {
                  return [{ name: currentFunctionName as string, args }];
                },
              } as unknown as GenerateContentResponse;
              currentFunctionName = undefined;
              currentFunctionArgsStr = '';
            }
            break;
          case 'message_delta':
            if ((chunk.delta as { stop_reason?: string }).stop_reason) {
              let finishReason = 'STOP';
              if (
                (chunk.delta as { stop_reason?: string }).stop_reason ===
                'max_tokens'
              ) {
                finishReason = 'MAX_TOKENS';
              }
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
          case 'message_stop':
            // stream done
            break;
        }
      }
    })();
  }

  private mapAnthropicResponseToGemini(
    response: Message,
  ): GenerateContentResponse {
    const parts: Part[] = [];

    for (const block of response.content) {
      if (block.type === 'text') {
        parts.push({ text: block.text });
      } else if (block.type === 'tool_use') {
        parts.push({
          functionCall: {
            // we augment id onto the functionCall Object here so it returns when tool_result answers
            id: block.id,
            name: block.name,
            args: block.input as Record<string, unknown>,
          } as import('@google/genai').FunctionCall, // Satisfy types but still inject ID
        });
      }
    }

    return {
      candidates: [
        {
          content: {
            role: 'model',
            parts: parts,
          },
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

  async countTokens(
    request: CountTokensParameters,
  ): Promise<CountTokensResponse> {
    let textStr = '';
    if (Array.isArray(request.contents)) {
      textStr = request.contents
        .map((c) =>
          (c as Content).parts
            ? ((c as Content).parts as Part[])
                .map((p: Part) => p.text || '')
                .join(' ')
            : '',
        )
        .join('\n');
    } else {
      textStr =
        typeof request.contents === 'string'
          ? request.contents
          : JSON.stringify(request.contents);
    }

    const count = Math.ceil(textStr.length / 4);

    return {
      totalTokens: count,
    };
  }

  async embedContent(
    _request: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    throw new Error(
      'Embeddings not supported natively via Anthropic Vertex SDK',
    );
  }
}
