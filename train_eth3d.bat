@echo off
REM ============================================================================
REM ETH3D 多精度量化训练启动脚本
REM ============================================================================
REM
REM 这个脚本用于启动不同精度的训练实验
REM
REM 使用方法：
REM   train_eth3d.bat [模式] [GPU数量]
REM
REM 模式选项：
REM   fp32      - FP32 Baseline训练
REM   int8      - INT8量化训练
REM   int4      - INT4量化训练
REM   all       - 运行所有实验（默认）
REM
REM 示例：
REM   train_eth3d.bat fp32 1
REM   train_eth3d.bat int8 2
REM   train_eth3d.bat all 1
REM
REM ============================================================================

setlocal enabledelayedexpansion

REM 设置默认参数
set MODE=%1
set NUM_GPUS=%2

if "%MODE%"=="" set MODE=all
if "%NUM_GPUS%"=="" set NUM_GPUS=1

echo ============================================================================
echo           ETH3D 多精度量化训练
echo ============================================================================
echo 模式: %MODE%
echo GPU数量: %NUM_GPUS%
echo ============================================================================
echo.

REM 检查环境
python --version >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] Python未安装
    pause
    exit /b 1
)

REM 进入训练目录
cd vggt\training

REM 根据模式运行训练
if "%MODE%"=="fp32" (
    call :train_fp32
) else if "%MODE%"=="int8" (
    call :train_int8
) else if "%MODE%"=="int4" (
    call :train_int4
) else if "%MODE%"=="all" (
    echo [INFO] 运行所有实验...
    call :train_fp32
    if %errorLevel% neq 0 goto :error
    call :train_int8
    if %errorLevel% neq 0 goto :error
    call :train_int4
    if %errorLevel% neq 0 goto :error
    echo [OK] 所有实验完成！
) else (
    echo [ERROR] 未知模式: %MODE%
    echo 可用模式: fp32, int8, int4, all
    pause
    exit /b 1
)

cd ..\..
echo.
echo ============================================================================
echo                         训练完成！
echo ============================================================================
echo.
echo 查看日志：cd vggt\logs
echo 查看TensorBoard：tensorboard --logdir=vggt\logs
echo.
pause
exit /b 0

REM ============================================================================
REM 训练函数
REM ============================================================================

:train_fp32
echo.
echo ============================================================================
echo 实验 1/3: FP32 Baseline训练
echo ============================================================================
echo.
echo [INFO] 配置: FP32完整精度
echo [INFO] 说明: 这是baseline实验，用于对比量化效果
echo.

REM 修改配置文件
python -c "import yaml; config = yaml.safe_load(open('config/eth3d_quantization.yaml')); config['quantization']['current_scheme'] = 'fp32'; config['exp_name'] = 'eth3d_fp32_baseline'; yaml.dump(config, open('config/eth3d_quantization.yaml', 'w'))"

REM 启动训练
torchrun --nproc_per_node=%NUM_GPUS% launch.py --config eth3d_quantization

if %errorLevel% neq 0 (
    echo [ERROR] FP32训练失败
    exit /b 1
)

echo [OK] FP32 Baseline训练完成
goto :eof

:train_int8
echo.
echo ============================================================================
echo 实验 2/3: INT8 Per-Channel量化训练
echo ============================================================================
echo.
echo [INFO] 配置: INT8 Per-Channel Symmetric
echo [INFO] 说明: 逐通道对称量化，平衡精度和压缩率
echo.

REM 修改配置文件
python -c "import yaml; config = yaml.safe_load(open('config/eth3d_quantization.yaml')); config['quantization']['current_scheme'] = 'int8_per_channel_sym'; config['quantization']['bits'] = 8; config['quantization']['per_channel'] = True; config['exp_name'] = 'eth3d_int8_perchannel'; yaml.dump(config, open('config/eth3d_quantization.yaml', 'w'))"

REM 启动训练
torchrun --nproc_per_node=%NUM_GPUS% launch.py --config eth3d_quantization

if %errorLevel% neq 0 (
    echo [ERROR] INT8训练失败
    exit /b 1
)

echo [OK] INT8量化训练完成
goto :eof

:train_int4
echo.
echo ============================================================================
echo 实验 3/3: INT4 Group-wise量化训练
echo ============================================================================
echo.
echo [INFO] 配置: INT4 Group-wise (Group Size=128)
echo [INFO] 说明: 分组量化，极致压缩
echo.

REM 修改配置文件
python -c "import yaml; config = yaml.safe_load(open('config/eth3d_quantization.yaml')); config['quantization']['current_scheme'] = 'int4_group_128'; config['quantization']['bits'] = 4; config['quantization']['group_size'] = 128; config['exp_name'] = 'eth3d_int4_group128'; yaml.dump(config, open('config/eth3d_quantization.yaml', 'w'))"

REM 启动训练
torchrun --nproc_per_node=%NUM_GPUS% launch.py --config eth3d_quantization

if %errorLevel% neq 0 (
    echo [ERROR] INT4训练失败
    exit /b 1
)

echo [OK] INT4量化训练完成
goto :eof

:error
echo [ERROR] 训练过程中出现错误
cd ..\..
pause
exit /b 1
