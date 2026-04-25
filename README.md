# Eino

A fork of [cloudwego/eino](https://github.com/cloudwego/eino) — a powerful LLM application development framework for Go.

## Overview

Eino provides a composable, type-safe framework for building LLM-powered applications in Go. It offers:

- **Graph-based orchestration**: Build complex AI pipelines using directed graphs
- **Type-safe components**: Strongly typed interfaces for LLM, retriever, tool, and other components
- **Streaming support**: First-class support for streaming responses
- **Extensible architecture**: Easy to integrate custom components and providers

## Features

- 🔗 Composable component graph execution
- 🌊 Native streaming with `io.Reader`-compatible interfaces
- 🔒 Type-safe component contracts
- 🛠️ Built-in components: ChatModel, Retriever, Tool, Lambda, etc.
- 📦 Modular design — use only what you need
- 🔄 Supports parallel and sequential execution patterns

## Installation

```bash
go get github.com/your-org/eino
```

## Quick Start

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/your-org/eino/compose"
    "github.com/your-org/eino/schema"
)

func main() {
    ctx := context.Background()

    // Build a simple chain
    chain, err := compose.NewChain[string, string]().
        AppendLambda(compose.InvokableLambda(func(ctx context.Context, input string) (string, error) {
            return "Hello, " + input + "!", nil
        })).
        Compile(ctx)
    if err != nil {
        log.Fatal(err)
    }

    result, err := chain.Invoke(ctx, "World")
    if err != nil {
        log.Fatal(err)
    }

    fmt.Println(result) // Hello, World!
    _ = schema.Message{} // schema is available
}
```

## Project Structure

```
eino/
├── compose/        # Graph and chain composition engine
├── components/     # Built-in component interfaces
│   ├── model/      # LLM chat model interface
│   ├── retriever/  # Document retrieval interface
│   ├── tool/       # Tool/function calling interface
│   └── ...
├── schema/         # Core data types (Message, Document, etc.)
├── flow/           # Pre-built flow patterns
└── utils/          # Utility helpers
```

## Personal Notes

> **Note (personal fork):** I'm using this fork primarily to experiment with custom `Retriever` implementations and RAG pipeline patterns. The upstream repo moves fast — I'll periodically sync from `cloudwego/eino:main`.
>
> **Current experiments:**
> - Custom BM25 + vector hybrid retriever in `components/retriever/hybrid/`
> - Testing chunk overlap strategies for long-document RAG
> - Benchmarking graph vs. chain performance for simple sequential pipelines
>
> **Sync log:**
> - 2025-07-02: Synced with upstream `cloudwego/eino@main` (commit `d7b04e8`)
> - 2025-06-10: Synced with upstream `cloudwego/eino@main` (commit `a3f91c2`)

## Contributing

We welcome contributions! Please see our [Pull Request Template](.github/PULL_REQUEST_TEMPLATE.md) and follow the commit conventions described in [`.github/.commit-rules.json`](.github/.commit-rules.json).

1. Fork the repository
2. Create your feature branch (`git check
