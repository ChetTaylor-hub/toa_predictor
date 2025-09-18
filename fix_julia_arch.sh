#!/bin/bash
# Julia架构修复脚本 - 专为Apple Silicon Mac设计

echo "=== Julia Architecture Fix for Apple Silicon Mac ==="
echo "Current system architecture: $(uname -m)"
echo "Current date: $(date)"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 检查当前Julia安装
check_current_julia() {
    echo -e "${BLUE}检查当前Julia安装...${NC}"
    
    if command -v julia &> /dev/null; then
        echo -e "${YELLOW}发现Julia安装:${NC}"
        julia --version
        
        # 检查Julia架构
        julia_arch=$(julia -e "println(Sys.MACHINE)" 2>/dev/null)
        echo "Julia架构: $julia_arch"
        
        # 检查系统架构
        system_arch=$(uname -m)
        echo "系统架构: $system_arch"
        
        if [[ "$julia_arch" == *"aarch64"* ]] && [[ "$system_arch" == "arm64" ]]; then
            echo -e "${GREEN}✓ Julia架构匹配 (ARM64)${NC}"
            return 0
        elif [[ "$julia_arch" == *"x86_64"* ]] && [[ "$system_arch" == "arm64" ]]; then
            echo -e "${RED}✗ 架构不匹配: Julia是x86_64，系统是ARM64${NC}"
            echo -e "${YELLOW}需要卸载x86_64版本并安装ARM64版本${NC}"
            return 1
        else
            echo -e "${RED}✗ 未知的架构组合${NC}"
            return 1
        fi
    else
        echo -e "${YELLOW}未找到Julia安装${NC}"
        return 2
    fi
}

# 卸载当前Julia
uninstall_julia() {
    echo -e "${BLUE}卸载当前Julia安装...${NC}"
    
    # 查找Julia安装位置
    julia_path=$(which julia 2>/dev/null)
    if [ -n "$julia_path" ]; then
        echo "找到Julia路径: $julia_path"
        
        # 如果是符号链接，找到实际位置
        if [ -L "$julia_path" ]; then
            real_path=$(readlink "$julia_path")
            echo "符号链接指向: $real_path"
        fi
        
        # 常见的Julia安装位置
        common_paths=(
            "/usr/local/bin/julia"
            "/opt/julia"
            "/Applications/Julia-*.app"
            "$HOME/.julia"
            "/usr/local/julia"
        )
        
        echo -e "${YELLOW}请手动删除以下Julia相关文件和目录:${NC}"
        for path in "${common_paths[@]}"; do
            if [ -e "$path" ] || [ -L "$path" ]; then
                echo "  - $path"
            fi
        done
        
        echo ""
        echo -e "${YELLOW}建议执行以下命令清理Julia:${NC}"
        echo "sudo rm -f /usr/local/bin/julia"
        echo "sudo rm -rf /opt/julia*"
        echo "rm -rf ~/.julia"
        echo "rm -rf /Applications/Julia-*.app"
        
        read -p "是否现在执行清理? (y/N): " confirm
        if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
            echo "清理Julia安装..."
            sudo rm -f /usr/local/bin/julia
            sudo rm -rf /opt/julia*
            rm -rf ~/.julia
            rm -rf /Applications/Julia-*.app 2>/dev/null
            echo -e "${GREEN}清理完成${NC}"
        else
            echo -e "${YELLOW}请手动清理后再运行此脚本${NC}"
            return 1
        fi
    fi
}

# 下载并安装ARM64版本的Julia
install_julia_arm64() {
    echo -e "${BLUE}下载并安装Julia ARM64版本...${NC}"
    
    # Julia版本信息
    JULIA_VERSION="1.10.10"
    JULIA_ARCH="macaarch64"
    JULIA_TAR="julia-${JULIA_VERSION}-${JULIA_ARCH}.tar.gz"
    JULIA_URL="https://julialang-s3.julialang.org/bin/mac/aarch64/1.10/${JULIA_TAR}"
    
    echo "版本: $JULIA_VERSION"
    echo "架构: ARM64 (aarch64)"
    echo "下载URL: $JULIA_URL"
    
    # 创建临时目录
    TEMP_DIR="/tmp/julia_install_$(date +%s)"
    mkdir -p "$TEMP_DIR"
    cd "$TEMP_DIR"
    
    echo -e "${BLUE}下载Julia...${NC}"
    if command -v curl &> /dev/null; then
        curl -L --progress-bar -o "$JULIA_TAR" "$JULIA_URL"
    elif command -v wget &> /dev/null; then
        wget --progress=bar -O "$JULIA_TAR" "$JULIA_URL"
    else
        echo -e "${RED}错误: 未找到curl或wget，请安装其中之一${NC}"
        return 1
    fi
    
    # 检查下载是否成功
    if [ ! -f "$JULIA_TAR" ] || [ ! -s "$JULIA_TAR" ]; then
        echo -e "${RED}错误: Julia下载失败${NC}"
        return 1
    fi
    
    echo -e "${GREEN}下载完成，文件大小: $(ls -lh $JULIA_TAR | awk '{print $5}')${NC}"
    
    # 解压Julia
    echo -e "${BLUE}解压Julia...${NC}"
    tar -xzf "$JULIA_TAR"
    
    if [ ! -d "julia-${JULIA_VERSION}" ]; then
        echo -e "${RED}错误: 解压失败${NC}"
        return 1
    fi
    
    # 安装到系统目录
    JULIA_INSTALL_DIR="/opt/julia-${JULIA_VERSION}"
    echo -e "${BLUE}安装Julia到 $JULIA_INSTALL_DIR...${NC}"
    
    sudo mkdir -p "$JULIA_INSTALL_DIR"
    sudo cp -R julia-${JULIA_VERSION}/* "$JULIA_INSTALL_DIR/"
    
    # 创建符号链接
    sudo ln -sf "$JULIA_INSTALL_DIR/bin/julia" /usr/local/bin/julia
    
    # 清理临时文件
    cd /
    rm -rf "$TEMP_DIR"
    
    echo -e "${GREEN}Julia安装完成!${NC}"
    
    # 验证安装
    if command -v julia &> /dev/null; then
        echo -e "${BLUE}验证安装:${NC}"
        julia --version
        julia -e "println(\"Julia架构: \", Sys.MACHINE)"
        echo -e "${GREEN}✓ Julia ARM64版本安装成功${NC}"
        return 0
    else
        echo -e "${RED}✗ Julia安装验证失败${NC}"
        return 1
    fi
}

# 配置Julia环境
setup_julia_environment() {
    echo -e "${BLUE}配置Julia环境...${NC}"
    
    # 设置环境变量
    JULIA_PATH=$(which julia)
    if [ -n "$JULIA_PATH" ]; then
        JULIA_HOME=$(dirname $(dirname $JULIA_PATH))
        export JULIA_BINDIR="$JULIA_HOME/bin"
        export JULIA_PKG_DEVDIR="$HOME/.julia/dev"
        
        echo "Julia路径: $JULIA_PATH"
        echo "Julia主目录: $JULIA_HOME"
        echo "二进制目录: $JULIA_BINDIR"
        echo "包开发目录: $JULIA_PKG_DEVDIR"
        
        # 将环境变量添加到shell配置文件
        SHELL_CONFIG=""
        if [ -n "$ZSH_VERSION" ]; then
            SHELL_CONFIG="$HOME/.zshrc"
        elif [ -n "$BASH_VERSION" ]; then
            SHELL_CONFIG="$HOME/.bash_profile"
        fi
        
        if [ -n "$SHELL_CONFIG" ]; then
            echo -e "${BLUE}添加环境变量到 $SHELL_CONFIG${NC}"
            echo "" >> "$SHELL_CONFIG"
            echo "# Julia环境变量 (添加时间: $(date))" >> "$SHELL_CONFIG"
            echo "export JULIA_BINDIR=\"$JULIA_BINDIR\"" >> "$SHELL_CONFIG"
            echo "export JULIA_PKG_DEVDIR=\"$JULIA_PKG_DEVDIR\"" >> "$SHELL_CONFIG"
        fi
        
        return 0
    else
        echo -e "${RED}错误: 未找到Julia${NC}"
        return 1
    fi
}

# 安装必要的Julia包
install_julia_packages() {
    echo -e "${BLUE}安装Julia包...${NC}"
    
    julia -e '
    using Pkg
    
    println("配置Julia包管理器...")
    
    # 添加注册表
    try
        println("添加General注册表...")
        Pkg.Registry.add("General")
    catch e
        println("General注册表已存在或出错: ", e)
    end
    
    # 要安装的包列表
    packages = [
        "UnderwaterAcoustics",
        "AcousticsToolbox", 
        "PythonCall",
        "Plots",
        "WAV",
        "DSP"
    ]
    
    println("开始安装包...")
    for pkg in packages
        println("正在安装: ", pkg)
        try
            Pkg.add(pkg)
            println("✓ 成功安装 ", pkg)
        catch e
            println("✗ 安装失败 ", pkg, ": ", e)
            # 继续安装其他包
        end
    end
    
    println("预编译包...")
    try
        Pkg.precompile()
        println("✓ 预编译完成")
    catch e
        println("预编译出错: ", e)
    end
    
    println("Julia包安装完成!")
    ' || {
        echo -e "${RED}Julia包安装过程中出现错误${NC}"
        return 1
    }
}

# 测试Julia安装
test_julia_installation() {
    echo -e "${BLUE}测试Julia安装...${NC}"
    
    # 基本测试
    echo "1. 基本功能测试:"
    julia -e 'println("Hello from Julia ARM64!")' || {
        echo -e "${RED}基本测试失败${NC}"
        return 1
    }
    
    # 架构测试
    echo "2. 架构测试:"
    julia -e '
    println("Julia版本: ", VERSION)
    println("系统架构: ", Sys.MACHINE)
    println("CPU核心数: ", Sys.CPU_THREADS)
    println("内存信息: ", round(Sys.total_memory()/1024^3, digits=2), " GB")
    ' || {
        echo -e "${RED}架构测试失败${NC}"
        return 1
    }
    
    # 包加载测试
    echo "3. 包加载测试:"
    julia -e '
    try
        using UnderwaterAcoustics
        println("✓ UnderwaterAcoustics 加载成功")
    catch e
        println("✗ UnderwaterAcoustics 加载失败: ", e)
    end
    
    try
        using PythonCall
        println("✓ PythonCall 加载成功")
    catch e
        println("✗ PythonCall 加载失败: ", e)
    end
    ' || {
        echo -e "${YELLOW}某些包加载失败，但Julia基本功能正常${NC}"
    }
    
    echo -e "${GREEN}Julia测试完成${NC}"
}

# 主函数
main() {
    echo -e "${BLUE}===================================================${NC}"
    echo -e "${BLUE}    Julia架构修复脚本 for Apple Silicon Mac     ${NC}"
    echo -e "${BLUE}===================================================${NC}"
    echo ""
    
    # 检查当前状态
    check_result=$(check_current_julia)
    check_status=$?
    
    case $check_status in
        0)
            echo -e "${GREEN}Julia已正确安装并匹配系统架构!${NC}"
            echo "是否仍要重新安装? (y/N): "
            read -r reinstall
            if [[ ! $reinstall == [yY] ]]; then
                echo "跳过重新安装"
                setup_julia_environment
                test_julia_installation
                exit 0
            fi
            ;;
        1)
            echo -e "${YELLOW}检测到架构不匹配，需要重新安装${NC}"
            ;;
        2)
            echo -e "${YELLOW}未检测到Julia，将进行全新安装${NC}"
            ;;
    esac
    
    # 如果需要重新安装，先卸载
    if [ $check_status -eq 1 ]; then
        uninstall_julia || {
            echo -e "${RED}卸载失败，请手动清理后重试${NC}"
            exit 1
        }
    fi
    
    # 安装ARM64版本
    install_julia_arm64 || {
        echo -e "${RED}Julia安装失败${NC}"
        exit 1
    }
    
    # 配置环境
    setup_julia_environment || {
        echo -e "${RED}环境配置失败${NC}"
        exit 1
    }
    
    # 安装包
    install_julia_packages || {
        echo -e "${YELLOW}包安装过程中有错误，但可能不影响基本使用${NC}"
    }
    
    # 测试安装
    test_julia_installation
    
    echo ""
    echo -e "${GREEN}===================================================${NC}"
    echo -e "${GREEN}           Julia架构修复完成!                    ${NC}"
    echo -e "${GREEN}===================================================${NC}"
    echo ""
    echo -e "${BLUE}现在您可以使用正确的ARM64版本的Julia了${NC}"
    echo -e "${BLUE}请重新启动终端或运行 'source ~/.zshrc' 来加载新的环境变量${NC}"
    echo ""
    echo -e "${YELLOW}测试命令:${NC}"
    echo "  julia --version"
    echo "  julia -e 'println(Sys.MACHINE)'"
    echo ""
}

# 检查是否在macOS上运行
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo -e "${RED}此脚本仅适用于macOS系统${NC}"
    exit 1
fi

# 检查是否为Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo -e "${RED}此脚本仅适用于Apple Silicon Mac${NC}"
    exit 1
fi

# 运行主函数
main "$@"
