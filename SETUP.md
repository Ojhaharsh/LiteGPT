# Setup Instructions

This project builds and runs on Windows with MSVC (C++17).

## Prerequisites

- Visual Studio Build Tools (or Visual Studio Community) with **Desktop development with C++**
- Git
- (Optional) Python 3.8+ for the weight downloader

## 1) Clone

```bat
git clone https://github.com/Ojhaharsh/LiteGPT.git
cd LiteGPT
```

## 2) Get model files (required for real generation)

Model files are intentionally ignored by Git (see `.gitignore`). Download them locally.

### Option A (recommended): one command

```bat
get_weights.bat
```

### Option B: run the downloader directly

```bat
python download_weights.py
```

This creates:
- `models/gpt2_weights.bin`
- `models/gpt2_config.json`
- `models/vocab.json`

### Important: copy files to the project root

The current executable looks for these filenames in the **project root** by default:
- `gpt2_weights.bin`
- `gpt2_config.json`
- `vocab.json`

Copy them once after download:

```bat
copy models\gpt2_weights.bin .
copy models\gpt2_config.json .
copy models\vocab.json .
```

## 3) Build

### Option A: build + run (outputs `llm_engine.exe` in root)

```bat
compile.bat
```

### Option B: build + run (outputs to `build/`)

```bat
build.bat
```

## 4) Run

If you built with `compile.bat`:

```bat
llm_engine.exe
```

If you built with `build.bat`:

```bat
build\llm_engine.exe
```

## 5) Quick smoke tests (non-interactive)

You can provide a single prompt via stdin to quickly verify generation without typing in the interactive loop:

If you built with `compile.bat`:

```bat
echo hello world | llm_engine.exe
echo once upon a time | llm_engine.exe
```

If you built with `build.bat`:

```bat
echo hello world | build\llm_engine.exe
echo once upon a time | build\llm_engine.exe
```

## Optional: CMake build

```bat
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022"
cmake --build . --config Release
Release\llm_engine.exe
```

## Expected Output

You should see an interactive prompt that generates continuations once the model files are present.
