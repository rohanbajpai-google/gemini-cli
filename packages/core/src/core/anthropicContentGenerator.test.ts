/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { AnthropicContentGenerator } from './anthropicContentGenerator.js';
import type { Message } from '@anthropic-ai/sdk/resources/messages';
import type { GenerateContentResponse } from '@google/genai';

const mockCreate = vi.fn();
const mockStream = vi.fn();

vi.mock('@anthropic-ai/vertex-sdk', () => ({
  AnthropicVertex: class {
    messages = {
      create: mockCreate,
      stream: mockStream,
    };
  },
}));

/** Helper to access private methods for unit-testing internal mappings. */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type PrivateGenerator = any;

describe('AnthropicContentGenerator', () => {
  let generator: AnthropicContentGenerator;

  beforeEach(() => {
    vi.clearAllMocks();
    generator = new AnthropicContentGenerator('test-project', 'us-central1');
  });

  describe('mapContentsToMessages', () => {
    it('should map a simple text user message', () => {
      const contents = [{ role: 'user', parts: [{ text: 'Hello Claude' }] }];

      const messages = (generator as PrivateGenerator).mapContentsToMessages(
        contents,
      );

      expect(messages).toEqual([
        {
          role: 'user',
          content: [{ type: 'text', text: 'Hello Claude' }],
        },
      ]);
    });

    it('should map model role to assistant', () => {
      const contents = [{ role: 'model', parts: [{ text: 'Hello!' }] }];

      const messages = (generator as PrivateGenerator).mapContentsToMessages(
        contents,
      );

      expect(messages).toHaveLength(1);
      expect(messages[0].role).toBe('assistant');
    });

    it('should map tool use parts correctly', () => {
      const contents = [
        {
          role: 'model',
          parts: [
            {
              functionCall: {
                name: 'get_weather',
                args: { location: 'San Francisco' },
              },
            },
          ],
        },
      ];

      const messages = (generator as PrivateGenerator).mapContentsToMessages(
        contents,
      );

      // The assistant message plus a synthetic user message with tool_result
      expect(messages).toHaveLength(2);
      expect(messages[0].role).toBe('assistant');
      const block = messages[0].content[0] as {
        type: string;
        name: string;
        input: Record<string, unknown>;
        id?: string;
      };
      expect(block.type).toBe('tool_use');
      expect(block.name).toBe('get_weather');
      expect(block.input).toEqual({ location: 'San Francisco' });

      // Synthetic tool_result follows
      expect(messages[1].role).toBe('user');
      const resultBlock = (
        messages[1].content as Array<{ type: string; tool_use_id: string }>
      )[0];
      expect(resultBlock.type).toBe('tool_result');
    });

    it('should map function response parts to tool_result blocks', () => {
      const contents = [
        {
          role: 'user',
          parts: [
            {
              functionResponse: {
                id: 'toolu_abc',
                name: 'get_weather',
                response: { temperature: 72 },
              },
            },
          ],
        },
      ];

      const messages = (generator as PrivateGenerator).mapContentsToMessages(
        contents,
      );

      expect(messages).toHaveLength(1);
      const block = messages[0].content[0] as {
        type: string;
        tool_use_id: string;
        content: string;
      };
      expect(block.type).toBe('tool_result');
      expect(block.tool_use_id).toBe('toolu_abc');
      expect(block.content).toBe('{"temperature":72}');
    });

    it('should handle inline image data', () => {
      const contents = [
        {
          role: 'user',
          parts: [
            {
              inlineData: {
                mimeType: 'image/png',
                data: 'base64data',
              },
            },
          ],
        },
      ];

      const messages = (generator as PrivateGenerator).mapContentsToMessages(
        contents,
      );

      const block = messages[0].content[0] as {
        type: string;
        source: { type: string; media_type: string; data: string };
      };
      expect(block.type).toBe('image');
      expect(block.source.media_type).toBe('image/png');
      expect(block.source.data).toBe('base64data');
    });

    it('should handle a non-array ContentListUnion', () => {
      const contents = {
        role: 'user',
        parts: [{ text: 'single content' }],
      };

      const messages = (generator as PrivateGenerator).mapContentsToMessages(
        contents,
      );

      expect(messages).toHaveLength(1);
      expect(messages[0].content[0]).toEqual({
        type: 'text',
        text: 'single content',
      });
    });

    it('should merge consecutive user messages', () => {
      const contents = [
        { role: 'user', parts: [{ text: 'first user msg' }] },
        { role: 'user', parts: [{ text: 'second user msg' }] },
      ];

      const messages = (generator as PrivateGenerator).mapContentsToMessages(
        contents,
      );

      expect(messages).toHaveLength(1);
      expect(messages[0].role).toBe('user');
      expect(messages[0].content).toEqual([
        { type: 'text', text: 'first user msg' },
        { type: 'text', text: 'second user msg' },
      ]);
    });

    it('should merge consecutive model (assistant) messages', () => {
      const contents = [
        { role: 'user', parts: [{ text: 'hello' }] },
        { role: 'model', parts: [{ text: 'first response' }] },
        { role: 'model', parts: [{ text: 'continued response' }] },
      ];

      const messages = (generator as PrivateGenerator).mapContentsToMessages(
        contents,
      );

      expect(messages).toHaveLength(2);
      expect(messages[1].role).toBe('assistant');
      expect(messages[1].content).toEqual([
        { type: 'text', text: 'first response' },
        { type: 'text', text: 'continued response' },
      ]);
    });

    it('should inject synthetic tool_result when tool_use has no following tool_result', () => {
      const contents = [
        {
          role: 'model',
          parts: [
            {
              functionCall: {
                id: 'toolu_123',
                name: 'read_file',
                args: { path: '/tmp/test' },
              },
            },
          ],
        },
        // No user message with tool_result follows
        { role: 'user', parts: [{ text: 'next prompt' }] },
      ];

      const messages = (generator as PrivateGenerator).mapContentsToMessages(
        contents,
      );

      // The user message should now contain the synthetic tool_result
      // prepended before the original text content.
      expect(messages).toHaveLength(2);
      expect(messages[0].role).toBe('assistant');
      expect(messages[1].role).toBe('user');

      const userBlocks = messages[1].content as Array<{
        type: string;
        tool_use_id?: string;
        text?: string;
      }>;
      expect(userBlocks[0].type).toBe('tool_result');
      expect(userBlocks[0].tool_use_id).toBe('toolu_123');
      expect(userBlocks[1].type).toBe('text');
      expect(userBlocks[1].text).toBe('next prompt');
    });

    it('should inject synthetic user message with tool_result when tool_use is the last message', () => {
      const contents = [
        { role: 'user', parts: [{ text: 'do something' }] },
        {
          role: 'model',
          parts: [
            {
              functionCall: {
                id: 'toolu_456',
                name: 'run_shell',
                args: { command: 'ls' },
              },
            },
          ],
        },
      ];

      const messages = (generator as PrivateGenerator).mapContentsToMessages(
        contents,
      );

      expect(messages).toHaveLength(3);
      expect(messages[2].role).toBe('user');
      const syntheticBlock = (
        messages[2].content as Array<{
          type: string;
          tool_use_id: string;
          content: string;
        }>
      )[0];
      expect(syntheticBlock.type).toBe('tool_result');
      expect(syntheticBlock.tool_use_id).toBe('toolu_456');
    });

    it('should not inject synthetic tool_result when tool_result already exists', () => {
      const contents = [
        {
          role: 'model',
          parts: [
            {
              functionCall: {
                id: 'toolu_789',
                name: 'read_file',
                args: { path: '/tmp/test' },
              },
            },
          ],
        },
        {
          role: 'user',
          parts: [
            {
              functionResponse: {
                id: 'toolu_789',
                name: 'read_file',
                response: 'file contents',
              },
            },
          ],
        },
      ];

      const messages = (generator as PrivateGenerator).mapContentsToMessages(
        contents,
      );

      expect(messages).toHaveLength(2);
      // Only the real tool_result should be present
      const userBlocks = messages[1].content as Array<{
        type: string;
        tool_use_id?: string;
      }>;
      expect(userBlocks).toHaveLength(1);
      expect(userBlocks[0].type).toBe('tool_result');
      expect(userBlocks[0].tool_use_id).toBe('toolu_789');
    });

    it('should handle multiple tool_use blocks with partial tool_results', () => {
      const contents = [
        {
          role: 'model',
          parts: [
            {
              functionCall: {
                id: 'toolu_a',
                name: 'tool1',
                args: {},
              },
            },
            {
              functionCall: {
                id: 'toolu_b',
                name: 'tool2',
                args: {},
              },
            },
          ],
        },
        {
          role: 'user',
          parts: [
            {
              functionResponse: {
                id: 'toolu_a',
                name: 'tool1',
                response: 'result1',
              },
            },
            // toolu_b has no matching tool_result
          ],
        },
      ];

      const messages = (generator as PrivateGenerator).mapContentsToMessages(
        contents,
      );

      expect(messages).toHaveLength(2);
      const userBlocks = messages[1].content as Array<{
        type: string;
        tool_use_id?: string;
        content?: string;
      }>;
      // Should have synthetic tool_result for toolu_b prepended, plus the real one
      expect(userBlocks).toHaveLength(2);
      // Synthetic is prepended
      expect(userBlocks[0].type).toBe('tool_result');
      expect(userBlocks[0].tool_use_id).toBe('toolu_b');
      expect(userBlocks[0].content).toBe('No result available.');
      // Original follows
      expect(userBlocks[1].type).toBe('tool_result');
      expect(userBlocks[1].tool_use_id).toBe('toolu_a');
    });
  });

  describe('mapAnthropicResponseToGemini', () => {
    it('should map a text response to Gemini format', () => {
      const anthropicResponse = {
        content: [{ type: 'text', text: 'Here is your answer.' }],
        usage: { input_tokens: 10, output_tokens: 5 },
      } as unknown as Message;

      const response: GenerateContentResponse = (
        generator as PrivateGenerator
      ).mapAnthropicResponseToGemini(anthropicResponse);

      expect(response.candidates![0].content!.parts).toEqual([
        { text: 'Here is your answer.' },
      ]);
      expect(response.usageMetadata).toEqual({
        promptTokenCount: 10,
        candidatesTokenCount: 5,
        totalTokenCount: 15,
      });
      expect((response as unknown as { text: string }).text).toBe(
        'Here is your answer.',
      );
    });

    it('should map a tool use response to Gemini functionCall', () => {
      const anthropicResponse = {
        content: [
          { type: 'text', text: 'Let me check on that.' },
          {
            type: 'tool_use',
            id: 'toolu_123',
            name: 'search_web',
            input: { query: 'test query' },
          },
        ],
        usage: { input_tokens: 10, output_tokens: 20 },
      } as unknown as Message;

      const response: GenerateContentResponse = (
        generator as PrivateGenerator
      ).mapAnthropicResponseToGemini(anthropicResponse);

      expect(response.candidates![0].content!.parts).toEqual([
        { text: 'Let me check on that.' },
        {
          functionCall: {
            id: 'toolu_123',
            name: 'search_web',
            args: { query: 'test query' },
          },
        },
      ]);

      const typedResponse = response as unknown as {
        functionCalls: Array<{ name: string }>;
      };
      expect(typedResponse.functionCalls?.[0].name).toBe('search_web');
    });

    it('should return undefined functionCalls when response has no tool use', () => {
      const anthropicResponse = {
        content: [{ type: 'text', text: 'Just text.' }],
        usage: { input_tokens: 5, output_tokens: 3 },
      } as unknown as Message;

      const response: GenerateContentResponse = (
        generator as PrivateGenerator
      ).mapAnthropicResponseToGemini(anthropicResponse);

      const typedResponse = response as unknown as {
        functionCalls: undefined;
      };
      expect(typedResponse.functionCalls).toBeUndefined();
    });
  });

  describe('extractSystemPrompt', () => {
    it('should return undefined when no system instruction is set', () => {
      const result = (generator as PrivateGenerator).extractSystemPrompt({
        model: 'test',
        contents: [],
        config: {},
      });
      expect(result).toBeUndefined();
    });

    it('should return string system instructions as-is', () => {
      const result = (generator as PrivateGenerator).extractSystemPrompt({
        model: 'test',
        contents: [],
        config: { systemInstruction: 'You are a helpful assistant.' },
      });
      expect(result).toBe('You are a helpful assistant.');
    });

    it('should extract text from Content-style system instructions', () => {
      const result = (generator as PrivateGenerator).extractSystemPrompt({
        model: 'test',
        contents: [],
        config: {
          systemInstruction: {
            parts: [{ text: 'Line 1' }, { text: 'Line 2' }],
          },
        },
      });
      expect(result).toBe('Line 1\nLine 2');
    });
  });

  describe('mapToolDeclarations', () => {
    it('should return undefined for empty tools', () => {
      expect(
        (generator as PrivateGenerator).mapToolDeclarations([]),
      ).toBeUndefined();
      expect(
        (generator as PrivateGenerator).mapToolDeclarations(undefined),
      ).toBeUndefined();
    });

    it('should map function declarations to Anthropic tools', () => {
      const tools = [
        {
          functionDeclarations: [
            {
              name: 'read_file',
              description: 'Reads a file',
              parameters: {
                type: 'object',
                properties: { path: { type: 'string' } },
              },
            },
          ],
        },
      ];

      const result = (generator as PrivateGenerator).mapToolDeclarations(tools);

      expect(result).toHaveLength(1);
      expect(result[0].name).toBe('read_file');
      expect(result[0].description).toBe('Reads a file');
    });
  });

  describe('countTokens', () => {
    it('should estimate tokens from array contents', async () => {
      const result = await generator.countTokens({
        model: 'test',
        contents: [{ role: 'user', parts: [{ text: 'Hello world' }] }],
      });

      // "Hello world" = 11 chars, ceil(11/4) = 3
      expect(result.totalTokens).toBe(3);
    });

    it('should estimate tokens from string contents', async () => {
      const result = await generator.countTokens({
        model: 'test',
        contents: 'A short string',
      });

      // "A short string" = 14 chars, ceil(14/4) = 4
      expect(result.totalTokens).toBe(4);
    });
  });

  describe('embedContent', () => {
    it('should throw an error', async () => {
      await expect(
        generator.embedContent({ model: 'test', content: '' }),
      ).rejects.toThrow('Embeddings are not supported');
    });
  });

  describe('generateContent', () => {
    it('should call client.messages.create with correct params', async () => {
      const mockResponse: Message = {
        id: 'msg_123',
        type: 'message',
        role: 'assistant',
        model: 'claude-3-5-sonnet-v2@20241022',
        content: [{ type: 'text', text: 'Response text' }],
        usage: { input_tokens: 10, output_tokens: 5 },
        stop_reason: 'end_turn',
        stop_sequence: null,
      };
      mockCreate.mockResolvedValueOnce(mockResponse);

      const response = await generator.generateContent(
        {
          model: 'claude-3-5-sonnet-v2@20241022',
          contents: [{ role: 'user', parts: [{ text: 'Hello' }] }],
          config: { temperature: 0.5 },
        },
        'prompt-1',
        'primary' as never,
      );

      expect(mockCreate).toHaveBeenCalledOnce();
      const callArgs = mockCreate.mock.calls[0][0];
      expect(callArgs.model).toBe('claude-3-5-sonnet-v2@20241022');
      expect(callArgs.temperature).toBe(0.5);
      expect(response.candidates![0].content!.parts).toEqual([
        { text: 'Response text' },
      ]);
    });
  });
});
