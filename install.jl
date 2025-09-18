using Pkg

# 清理预编译缓存
try
    rm(joinpath(homedir(), ".julia", "compiled"), recursive=true, force=true)
catch
    # 忽略删除错误
end

# 安装包时指定版本以避免兼容性问题
println("正在安装Julia包...")

# 数据集生成所需的包列表
packages = ["Random", "Plots", "CSV", "DataFrames", "JSON", "UnderwaterAcoustics", "AcousticsToolbox", "FFTW"]

for pkg in packages
    println("正在安装包: $pkg")
    try
        Pkg.add(pkg)
        println("✓ $pkg 安装成功")
    catch e
        println("✗ $pkg 安装失败: $e")
    end
end

println("包安装完成！")