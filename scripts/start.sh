#!/bin/bash

# SemanticCoreDB 启动脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_message() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查 Python 版本
check_python_version() {
    print_message "检查 Python 版本..."
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    required_version="3.9"
    
    if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
        print_message "Python 版本满足要求: $python_version"
    else
        print_error "Python 版本过低，需要 3.9 或更高版本，当前版本: $python_version"
        exit 1
    fi
}

# 检查依赖
check_dependencies() {
    print_message "检查依赖..."
    
    # 检查 pip
    if ! command -v pip3 &> /dev/null; then
        print_error "pip3 未安装"
        exit 1
    fi
    
    # 检查虚拟环境
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        print_warning "建议在虚拟环境中运行"
    fi
}

# 安装依赖
install_dependencies() {
    print_message "安装 Python 依赖..."
    pip3 install -r requirements.txt
}

# 创建必要目录
create_directories() {
    print_message "创建必要目录..."
    mkdir -p data/events
    mkdir -p data/objects
    mkdir -p data/vectors
    mkdir -p data/metadata
    mkdir -p logs
    mkdir -p temp
}

# 检查环境变量
check_environment() {
    print_message "检查环境变量..."
    
    # 检查 OpenAI API Key
    if [[ -z "$OPENAI_API_KEY" ]]; then
        print_warning "OPENAI_API_KEY 未设置，某些功能可能不可用"
    else
        print_message "OpenAI API Key 已设置"
    fi
    
    # 检查其他环境变量
    if [[ -z "$SCDB_HOST" ]]; then
        export SCDB_HOST="localhost"
    fi
    
    if [[ -z "$SCDB_PORT" ]]; then
        export SCDB_PORT="8000"
    fi
}

# 启动数据库
start_database() {
    print_message "启动 SemanticCoreDB..."
    
    # 检查是否使用 Docker
    if [[ "$1" == "--docker" ]]; then
        print_message "使用 Docker 启动..."
        docker-compose up -d
    else
        print_message "使用本地 Python 启动..."
        python3 -m src.api.main
    fi
}

# 运行测试
run_tests() {
    print_message "运行测试..."
    python3 -m pytest tests/ -v
}

# 运行示例
run_examples() {
    print_message "运行示例..."
    python3 examples/basic_usage.py
}

# 显示帮助信息
show_help() {
    echo "SemanticCoreDB 启动脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --help, -h          显示此帮助信息"
    echo "  --install           安装依赖"
    echo "  --test              运行测试"
    echo "  --example           运行示例"
    echo "  --docker            使用 Docker 启动"
    echo "  --dev               开发模式启动"
    echo ""
    echo "示例:"
    echo "  $0                  # 正常启动"
    echo "  $0 --docker         # Docker 启动"
    echo "  $0 --test           # 运行测试"
    echo "  $0 --example        # 运行示例"
}

# 主函数
main() {
    print_message "SemanticCoreDB 启动脚本开始执行..."
    
    # 解析命令行参数
    case "$1" in
        --help|-h)
            show_help
            exit 0
            ;;
        --install)
            check_python_version
            check_dependencies
            install_dependencies
            create_directories
            print_message "安装完成"
            exit 0
            ;;
        --test)
            check_python_version
            check_dependencies
            create_directories
            run_tests
            exit 0
            ;;
        --example)
            check_python_version
            check_dependencies
            create_directories
            run_examples
            exit 0
            ;;
        --docker)
            check_environment
            start_database --docker
            ;;
        --dev)
            check_python_version
            check_dependencies
            create_directories
            check_environment
            print_message "开发模式启动..."
            export SCDB_DEBUG=true
            python3 -m src.api.main --reload
            ;;
        "")
            check_python_version
            check_dependencies
            create_directories
            check_environment
            start_database
            ;;
        *)
            print_error "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"