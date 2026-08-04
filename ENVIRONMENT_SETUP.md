# 选股组合投资模型 环境设置指南

## ✅ 已完成的设置

### 已安装的库
- ✓ yfinance 1.3.0 - Yahoo Finance数据接口
- ✓ pandas 2.3.3 - 数据处理
- ✓ numpy 2.3.5 - 数值计算
- ✓ requests 2.32.5 - HTTP请求
- ✓ beautifulsoup4 4.14.3 - HTML解析
- ✓ scipy 1.16.3 - 科学计算
- ✓ scikit-learn 1.7.2 - 机器学习库
- ✓ apscheduler 3.11.2 - **新增：定时任务调度** ⭐

### Python版本
```
Python 3.13.9
位置: /opt/anaconda3/bin/python3
```

## 📦 依赖项完整列表

所有必需库已保存到 `requirements.txt`

```
yfinance>=1.0
pandas>=2.0
numpy>=1.20
requests>=2.28
beautifulsoup4>=4.11
scipy>=1.8
scikit-learn>=1.0
apscheduler>=3.10
```

## 🚀 如何使用

### 方式1：一键安装所有依赖
```bash
pip install -r requirements.txt
```

### 方式2：单独安装缺失的库
```bash
pip install apscheduler
```

### 方式3：验证环境
```bash
python3 -c "
from scheduler import ScheduledRebalancer
print('✓ 环境配置正确')
"
```

## 🔧 可选的高级优化库

### PyPortfolioOpt（可选）
如果需要更高级的投资组合优化算法，可尝试：

```bash
# 方法1：使用PyPI（需要git）
pip install git+https://github.com/robertmartin8/PyPortfolioOpt

# 方法2：如果上述方法失败，系统会自动使用等权重备选方案
```

**注意：** 即使PyPortfolioOpt安装失败，代码也有备用方案（等权重分配），不会导致程序崩溃。

## ✨ 现在可以运行的脚本

### 1. 快速测试（推荐先试这个）
```bash
python3 quick_test.py
```
- 运行时间：~2-3分钟
- 用途：验证环境，20只样本股票测试

### 2. 完整主pipeline
```bash
python3 main_pipeline.py
```
- 运行时间：~30-45分钟（首次，会下载503只S&P成分股）
- 用途：完整选股分析（之后会缓存数据）

### 3. 自动化重平衡
```bash
python3 auto_rebalance.py
```
- 用途：手动执行一次完整的投资组合优化

### 4. 启动定时执行器
```bash
python3 scheduler.py
```
- 用途：在后台自动执行定时任务
- 功能：
  - 每月末工作日15:00执行月度重平衡
  - 每周一09:00检查相关性
  - 相关性升高时自动触发紧急重平衡

## 🐛 故障排除

### 问题：导入错误
**解决方案：**
```bash
pip install -r requirements.txt --upgrade
```

### 问题：网络超时
**解决方案：**
- 重新运行命令（Yahoo Finance API有时会超时）
- 数据会自动缓存，第二次运行会很快

### 问题：权限错误
**解决方案：**
```bash
pip install --user -r requirements.txt
```

## 📝 缓存机制

- 第一次运行：下载所有股票数据到 `cache/` 文件夹
- 第二次及之后：直接读取缓存（快速）
- 缓存文件格式：`weekly_data_YYYYMMDD_to_YYYYMMDD.pkl`

**清除缓存：**
```bash
rm -rf cache/*.pkl
```

## ✅ 验证安装

运行以下命令验证所有库都正确安装：

```bash
python3 << 'EOF'
import yfinance, pandas, numpy, requests
from bs4 import BeautifulSoup
import scipy, sklearn
from apscheduler.schedulers.background import BackgroundScheduler

print("✓ 所有库导入成功")
print("✓ 环境配置完成，可以开始使用")
EOF
```

---

**更新时间：** 2026-05-09
**环境：** Python 3.13.9
**状态：** ✅ 所有依赖已安装
