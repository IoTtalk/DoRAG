@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM Data Preprocessing Pipeline
REM Project root:
REM D:\Program Project\LLM\Master Thesis\DoRAG
REM ============================================================

REM ---------- Script directory ----------
set SCRIPT_DIR=2_Data_Preprocessing

REM ---------- Data directories ----------
set PDF_DIR=1_Enterprise_Data

set RAW_TEXT_DIR=test
set FLATTEN_TEXT_DIR=test2
set IDENTIFIER_DIR=test3
set CHUNK_DIR=test4
set REFINED_CHUNK_DIR=test5
set FINAL_DIR=test6

set FINAL_OUTPUT=%FINAL_DIR%\chunk.json

REM ---------- Model config ----------
set MODEL_NAME=meta-llama/Llama-3.2-3B-Instruct
set MODELS_DIR=models

REM ---------- Make sure output folder exists ----------
if not exist "%FINAL_DIR%" mkdir "%FINAL_DIR%"

echo.
echo ============================================================
echo [1/6] Extracting text from PDFs
echo ============================================================
python "%SCRIPT_DIR%\pdf_extraction.py" ^
    --pdf_dir "%PDF_DIR%" ^
    --output_dir "%RAW_TEXT_DIR%"

if errorlevel 1 (
    echo [ERROR] PDF extraction failed.
    exit /b 1
)

echo.
echo ============================================================
echo [2/6] Flattening extracted text
echo ============================================================
python "%SCRIPT_DIR%\text_flattener.py" ^
    --input_dir "%RAW_TEXT_DIR%" ^
    --output_dir "%FLATTEN_TEXT_DIR%"

if errorlevel 1 (
    echo [ERROR] Text flattening failed.
    exit /b 1
)

echo.
echo ============================================================
echo [3/6] Generating section identifiers
echo ============================================================
python "%SCRIPT_DIR%\identifier_generation.py" ^
    --input_dir "%FLATTEN_TEXT_DIR%" ^
    --output_dir "%IDENTIFIER_DIR%" ^
    --model_name "%MODEL_NAME%" ^
    --models_dir "%MODELS_DIR%" ^
    --done_file identifier_done.txt

if errorlevel 1 (
    echo [ERROR] Identifier generation failed.
    exit /b 1
)

echo.
echo ============================================================
echo [4/6] Extracting chunks by identifiers
echo ============================================================
python "%SCRIPT_DIR%\chunk_extraction.py" ^
    --identifier-dir "%IDENTIFIER_DIR%" ^
    --text-dir "%FLATTEN_TEXT_DIR%" ^
    --output-dir "%CHUNK_DIR%"

if errorlevel 1 (
    echo [ERROR] Chunk extraction failed.
    exit /b 1
)

echo.
echo ============================================================
echo [5/6] Refining chunks
echo ============================================================
python "%SCRIPT_DIR%\chunk_refinement.py" ^
    --chunk-dir "%CHUNK_DIR%" ^
    --output-dir "%REFINED_CHUNK_DIR%" ^
    --model "%MODEL_NAME%" ^
    --models-dir "%MODELS_DIR%"

if errorlevel 1 (
    echo [ERROR] Chunk refinement failed.
    exit /b 1
)

echo.
echo ============================================================
echo [6/6] Running final re-chunking
echo ============================================================
python "%SCRIPT_DIR%\re-chunking.py" ^
    --input_dir "%REFINED_CHUNK_DIR%" ^
    --output_file "%FINAL_OUTPUT%"

if errorlevel 1 (
    echo [ERROR] Re-chunking failed.
    exit /b 1
)

echo.
echo ============================================================
echo [SUCCESS] Data preprocessing pipeline completed.
echo Final output: %FINAL_OUTPUT%
echo ============================================================

endlocal