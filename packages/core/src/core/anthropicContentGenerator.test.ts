/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi } from 'vitest';
import { AnthropicContentGenerator } from './anthropicContentGenerator.js';
import type { Message } from '@anthropic-ai/sdk/resources/messages';

vi.mock('@anthropic-ai/vertex-sdk', () => ({
  AnthropicVertex: class {
    messages = {
      create: vi.fn(),
      stream: vi.fn(),
    };
  },
}));

describe('AnthropicContentGenerator', () => {
  const generator = new AnthropicContentGenerator(
    'test-project',
    'us-central1',
  );

  describe('mapGeminiContentsToAnthropicMessages', () => {
    it('should map a simple text user message', () => {
      const contents = [
        {
          role: 'user',
          parts: [{ text: 'Hello Claude' }],
        },
      ];

      // @ts-expect-error - testing private method
      const messages = generator.mapGeminiContentsToAnthropicMessages(contents);

      expect(messages).toEqual([
        {
          role: 'user',
          content: [{ type: 'text', text: 'Hello Claude' }],
        },
      ]);
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

      // @ts-expect-error - testing private method
      const messages = generator.mapGeminiContentsToAnthropicMessages(contents);

      expect(messages).toHaveLength(1);
      expect(messages[0].role).toBe('assistant');
      const content0 = messages[0].content[0] as {
        type: string;
        name: string;
        input: Record<string, unknown>;
        id?: string;
      };
      expect(content0.type).toBe('tool_use');
      expect(content0.name).toBe('get_weather');
      expect(content0.input).toEqual({ location: 'San Francisco' });
      expect(content0.id).toBeDefined();
    });
  });

  describe('mapAnthropicResponseToGemini', () => {
    it('should map Anthropic text response to Gemini format', () => {
      const anthropicResponse = {
        content: [{ type: 'text', text: 'Here is your answer.' }],
        usage: { input_tokens: 10, output_tokens: 5 },
      } as unknown as Message;

      // @ts-expect-error - testing private method
      const response =
        generator.mapAnthropicResponseToGemini(anthropicResponse);

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

    it('should map Anthropic tool use response to Gemini functionCall', () => {
      const anthropicResponse = {
        content: [
          { type: 'text', text: 'Let me check on that.' },
          {
            type: 'tool_use',
            id: 'toolu_123',
            name: 'search_web',
            input: { query: 'Anthropic SDK' },
          },
        ],
        usage: { input_tokens: 10, output_tokens: 20 },
      } as unknown as Message;

      // @ts-expect-error - testing private method
      const response =
        generator.mapAnthropicResponseToGemini(anthropicResponse);

      expect(response.candidates![0].content!.parts).toEqual([
        { text: 'Let me check on that.' },
        {
          functionCall: {
            name: 'search_web',
            args: { query: 'Anthropic SDK' },
          },
        },
      ]);
      expect(
        (response as unknown as { functionCalls: Array<{ name: string }> })
          .functionCalls?.[0].name,
      ).toBe('search_web');
    });
  });
});
