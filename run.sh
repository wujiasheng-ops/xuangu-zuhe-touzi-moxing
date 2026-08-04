#!/bin/bash

# Stock Analysis 快速启动脚本

cd "$(dirname "$0")"

echo "=========================================="
echo "Stock Analysis 项目启动"
echo "=========================================="
echo ""

# 检查Python环境
echo "✓ 检查Python环境..."
python3 --version

echo ""
echo "选择要运行的脚本："
echo "  1) 快速测试 (quick_test.py) - 2-3分钟"
echo "  2) 完整主程序 (main_pipeline.py) - 30-45分钟"
echo "  3) 自动重平衡 (auto_rebalance.py) - 30-45分钟"
echo "  4) 定时执行器 (scheduler.py) - 后台运行"
echo ""
read -p "请输入选择 (1-4): " choice

case $choice in
    1)
        echo ""
        echo "运行快速测试..."
        python3 quick_test.py
        ;;
    2)
        echo ""
        echo "运行完整主程序（首次运行会下载S&P 500数据，请耐心等待...）"
        python3 main_pipeline.py
        ;;
    3)
        echo ""
        echo "运行自动重平衡..."
        python3 auto_rebalance.py
        ;;
    4)
        echo ""
        echo "启动定时执行器..."
        python3 scheduler.py
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac
