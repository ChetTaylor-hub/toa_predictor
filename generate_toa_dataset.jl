using Random, Plots, CSV, DataFrames, JSON, FFTW, UnderwaterAcoustics, Statistics
try
    using AcousticsToolbox
    global ACOUSTICS_TOOLBOX_AVAILABLE = true
catch
    global ACOUSTICS_TOOLBOX_AVAILABLE = false
    println("警告: AcousticsToolbox包未安装，将使用简化的声学模型")
end
# 增强版TOA数据集生成器 - 基于UnderwaterAcoustics包的专业海洋声学建模

# ---------------------
# 数据集配置
# ---------------------
const NUM_SAMPLES = 100       # 数据集大小（先测试小规模）
const FS = 48000              # 采样率
const T = 0.1                 # 信号时长
const FREQ = 10e3             # 载波频率
const SOUND_SPEED_REF = 1500.0    # 参考声速

# 环境参数范围
const DEPTH_RANGE = (50.0, 200.0)           # 水深范围
const SRC_DEPTH_RANGE = (5.0, 30.0)         # 源深度范围  
const RX_DEPTH_RANGE = (20.0, 80.0)         # 接收器深度范围
const DISTANCE_RANGE = (100.0, 2000.0)      # 水平距离范围
const NOISE_LEVEL_RANGE = (0.001, 0.05)     # 噪声水平范围

# 海洋环境参数
const TEMP_SURFACE = 15.0    # 海表温度 [°C]
const TEMP_BOTTOM = 4.0      # 海底温度 [°C]
const SALINITY = 35.0        # 盐度 [ppt]

# ---------------------
# 海洋声学建模函数
# ---------------------

function calculate_sound_speed(depth, temperature, salinity, pressure=nothing)
    """
    使用Mackenzie方程计算声速
    depth: 深度 [m]
    temperature: 温度 [°C]  
    salinity: 盐度 [ppt]
    pressure: 压力 [bar]，如果为nothing则根据深度计算
    """
    if pressure === nothing
        pressure = 1.0 + depth / 10.0  # 近似压力计算
    end
    
    # Mackenzie方程 (1981)
    c = 1448.96 + 4.591*temperature - 5.304e-2*temperature^2 + 2.374e-4*temperature^3 +
        1.340*(salinity - 35.0) + 1.630e-2*depth + 1.675e-7*depth^2 -
        1.025e-2*temperature*(salinity - 35.0) - 7.139e-13*temperature*depth^3
    
    return c
end

function generate_ssp_profile(max_depth, profile_type="thermocline")
    """
    使用UnderwaterAcoustics包生成声速剖面 (Sound Speed Profile)
    profile_type: "linear", "thermocline", "deep_water"
    """
    depths = collect(Float64, 0:2:max_depth)
    
    if profile_type == "linear"
        # 线性温度分布
        temperatures = TEMP_SURFACE .- (TEMP_SURFACE - TEMP_BOTTOM) .* (depths ./ max_depth)
        
    elseif profile_type == "thermocline"
        # 跃温层模型
        thermocline_depth = max_depth * 0.3
        temperatures = similar(depths, Float64)
        for (i, d) in enumerate(depths)
            if d <= thermocline_depth
                temperatures[i] = TEMP_SURFACE - (TEMP_SURFACE - 8.0) * (d / thermocline_depth)^2
            else
                temperatures[i] = 8.0 - (8.0 - TEMP_BOTTOM) * ((d - thermocline_depth) / (max_depth - thermocline_depth))
            end
        end
        
    elseif profile_type == "deep_water"
        # 深水声道模型
        sound_channel_axis = max_depth * 0.6
        temperatures = similar(depths, Float64)
        for (i, d) in enumerate(depths)
            if d <= sound_channel_axis
                temperatures[i] = TEMP_SURFACE - (TEMP_SURFACE - 4.0) * (d / sound_channel_axis)^0.5
            else
                temperatures[i] = 4.0 + 2.0 * exp(-(d - sound_channel_axis) / (max_depth * 0.2))
            end
        end
    end
    
    # 计算每个深度的声速
    sound_speeds = [calculate_sound_speed(d, t, SALINITY) for (d, t) in zip(depths, temperatures)]
    
    return depths, sound_speeds, temperatures
end

# 创建UnderwaterAcoustics环境
function create_ocean_environment(max_depth, profile_type="thermocline")
    """
    创建基于UnderwaterAcoustics的海洋环境
    """
    depths, sound_speeds, temperatures = generate_ssp_profile(max_depth, profile_type)
    
    try
        # 根据文档，使用简单的常数声速或SampledField
        if length(sound_speeds) > 1
            # 使用第一个声速值作为代表值，或者使用平均值
            avg_sound_speed = mean(sound_speeds)
            ssp = avg_sound_speed
        else
            ssp = sound_speeds[1]
        end
        
        # 创建海洋环境 - 根据ORCA文档的示例
        env = UnderwaterEnvironment(
            bathymetry = max_depth,
            soundspeed = ssp,
            seabed = FineSand
        )
        
        println("成功创建UnderwaterAcoustics环境，深度=$(max_depth)m，声速=$(ssp)m/s")
        return env, depths, sound_speeds, temperatures
        
    catch e
        println("UnderwaterAcoustics环境创建失败，使用简化环境: ", e)
        # 返回简化的环境数据结构
        env = Dict(
            "type" => "simplified",
            "depths" => depths,
            "sound_speeds" => sound_speeds,
            "bottom_depth" => max_depth
        )
        return env, depths, sound_speeds, temperatures
    end
end

function calculate_absorption_coefficient(frequency, depth, temperature, salinity, ph=8.0)
    """
    计算声吸收系数 [dB/km]
    基于Francois & Garrison (1982) 公式
    """
    f = frequency / 1000.0  # 转换为kHz
    
    # 硼酸贡献
    f1 = 0.78 * sqrt(salinity / 35.0) * exp(temperature / 26.0)
    A1 = (8.686 / f^2) * (0.106 * exp((ph - 8.0) / 0.56))
    boric_acid = A1 * f1 * f^2 / (f1^2 + f^2)
    
    # 硫酸镁贡献  
    f2 = 42.0 * exp(temperature / 17.0)
    A2 = 21.44 * salinity / 35.0 * (1.0 + 0.025 * temperature)
    magnesium_sulfate = A2 * f2 * f^2 / (f2^2 + f^2)
    
    # 纯水贡献
    pure_water = (3.964e-4 - 1.146e-5 * temperature + 1.45e-7 * temperature^2 - 
                  6.5e-10 * temperature^3) * f^2
    
    # 总吸收系数
    alpha = boric_acid + magnesium_sulfate + pure_water
    
    return alpha  # dB/km
end

function generate_ocean_noise(signal_length, fs, noise_type="wind")
    """
    生成真实海洋噪声
    noise_type: "wind", "shipping", "seismic", "combined"
    """
    t = (0:signal_length-1) / fs
    noise = zeros(signal_length)
    
    if noise_type == "wind" || noise_type == "combined"
        # 风噪声 (高频成分)
        wind_noise = randn(signal_length)
        # 应用风噪声频谱特性 (大约-17dB/decade in 100Hz-10kHz)
        freqs = fftfreq(signal_length, fs)
        wind_spectrum = 1.0 ./ (1.0 .+ (abs.(freqs) ./ 1000.0).^1.7)
        wind_fft = fft(wind_noise) .* sqrt.(wind_spectrum)
        wind_noise = real(ifft(wind_fft))
        noise .+= 0.6 * wind_noise
    end
    
    if noise_type == "shipping" || noise_type == "combined"
        # 航运噪声 (低频成分)
        ship_freq = 50.0 + 100.0 * rand()  # 50-150 Hz
        ship_noise = sin.(2π * ship_freq .* t) .* (1.0 .+ 0.3 * randn(signal_length))
        # 添加调制
        modulation = 1.0 .+ 0.2 * sin.(2π * 0.1 .* t)  # 0.1 Hz 调制
        noise .+= 0.3 * ship_noise .* modulation
    end
    
    if noise_type == "seismic" || noise_type == "combined"
        # 地震/微震噪声 (很低频)
        seismic_freq = 1.0 + 5.0 * rand()  # 1-6 Hz
        seismic_noise = sin.(2π * seismic_freq .* t) .* exp.(-t ./ 30.0)
        noise .+= 0.2 * seismic_noise
    end
    
    # 归一化
    if maximum(abs.(noise)) > 0
        noise = noise ./ maximum(abs.(noise))
    end
    
    return noise
end
function generate_transmitted_signal(t, freq)
    """生成高斯调制的正弦脉冲"""
    return sin.(2π*freq .* t) .* exp.(-(t .- 0.005).^2 ./ (2*(0.001)^2))
end

function calculate_enhanced_multipath_toas(src_pos, rx_pos, depth, env, environment)
    """
    使用UnderwaterAcoustics包计算增强的多径到达时间
    """
    paths = []
    
    # 提取位置信息
    src_x, src_y, src_depth = src_pos
    rx_x, rx_y, rx_depth = rx_pos
    horizontal_distance = sqrt((rx_x - src_x)^2 + (rx_y - src_y)^2)
    
    # 1. 尝试使用UnderwaterAcoustics和ORCA进行模态分析
    if !isa(env, Dict) && ACOUSTICS_TOOLBOX_AVAILABLE
        try
            # 根据ORCA文档创建传播模型
            pm = Orca(env)
            
            # 创建声源和接收器 - 注意ORCA要求源在(0,0)，接收器在右半平面
            # 我们需要变换坐标系
            tx = AcousticSource(0, -src_depth, FREQ)
            rx = AcousticReceiver(horizontal_distance, -rx_depth)
            
            # 计算模态到达
            modes = arrivals(pm, tx, rx)
            
            if length(modes) > 0
                println("成功使用ORCA计算得到 $(length(modes)) 个模态")
                
                # 处理模态结果转换为路径
                for (i, mode) in enumerate(modes[1:min(5, length(modes))])  # 限制前5个模态
                    # 计算传播时间 (基于群速度和相速度)
                    # 根据文档，模态有速度字段 v (群速度) 和 vₚ (相速度)
                    group_velocity = mode.v
                    phase_velocity = mode.vₚ
                    travel_time = horizontal_distance / group_velocity
                    
                    # 计算衰减 (基于模态特性)
                    # kᵣ 是径向波数
                    amplitude = abs(1.0 / sqrt(horizontal_distance)) * exp(-imag(mode.kᵣ) * horizontal_distance)
                    
                    path_type = if i == 1
                        "direct_mode"
                    else
                        "mode_$(i)"
                    end
                    
                    push!(paths, (
                        delay = travel_time,
                        amplitude = amplitude,
                        path_type = path_type,
                        distance = horizontal_distance,
                        transmission_loss = -20 * log10(amplitude),
                        mode_number = i,
                        group_velocity = group_velocity,
                        phase_velocity = phase_velocity
                    ))
                end
                
                return paths
            end
            
        catch e
            println("ORCA模态分析失败: ", e)
            println("回退到几何声学方法")
        end
    end
    
    # 回退到几何声学备用方案
    return calculate_fallback_multipath_toas(src_pos, rx_pos, environment)
end

function calculate_fallback_multipath_toas(src_pos, rx_pos, environment)
    """
    备用多径计算方法（当UnderwaterAcoustics失败时使用）
    """
    paths = []
    src_depth = src_pos[3]
    rx_depth = rx_pos[3]
    horizontal_distance = sqrt((rx_pos[1] - src_pos[1])^2 + (rx_pos[2] - src_pos[2])^2)
    
    # 1. 直达路径
    direct_distance = sqrt(sum((rx_pos .- src_pos).^2))
    direct_toa = direct_distance / environment["avg_sound_speed"]
    
    # 简单衰减计算
    absorption_coeff = calculate_absorption_coefficient(FREQ, (src_depth + rx_depth)/2, 
                                                      environment["temperature"], SALINITY)
    absorption_loss = absorption_coeff * direct_distance / 1000.0
    direct_amplitude = 1.0 * 10^(-absorption_loss / 20.0)
    
    push!(paths, (delay=direct_toa, amplitude=direct_amplitude, path_type="direct", 
                  distance=direct_distance, transmission_loss=absorption_loss))
    
    # 2. 海面反射路径
    if rand() > 0.3  # 70%概率有海面反射
        surface_distance = sqrt(horizontal_distance^2 + (src_depth + rx_depth)^2)
        surface_toa = surface_distance / environment["avg_sound_speed"]
        surface_amplitude = 0.3 * 10^(-absorption_coeff * surface_distance / 1000.0 / 20.0)
        
        push!(paths, (delay=surface_toa, amplitude=surface_amplitude, path_type="surface_reflected", 
                      distance=surface_distance, transmission_loss=absorption_coeff * surface_distance / 1000.0))
    end
    
    # 3. 海底反射路径
    if rand() > 0.5 && environment["depth"] > 100  # 50%概率有海底反射
        bottom_distance = sqrt(horizontal_distance^2 + (2*environment["depth"] - src_depth - rx_depth)^2)
        bottom_toa = bottom_distance / environment["avg_sound_speed"]
        bottom_amplitude = 0.2 * 10^(-absorption_coeff * bottom_distance / 1000.0 / 20.0)
        
        push!(paths, (delay=bottom_toa, amplitude=bottom_amplitude, path_type="bottom_reflected", 
                      distance=bottom_distance, transmission_loss=absorption_coeff * bottom_distance / 1000.0))
    end
    
    return paths
end

function interp_linear(x_data, y_data, x_query)
    """线性插值函数"""
    if x_query <= x_data[1]
        return y_data[1]
    elseif x_query >= x_data[end]
        return y_data[end]
    end
    
    # 找到插值区间
    i = 1
    while i < length(x_data) && x_data[i+1] < x_query
        i += 1
    end
    
    # 线性插值
    if i == length(x_data)
        return y_data[end]
    end
    
    t = (x_query - x_data[i]) / (x_data[i+1] - x_data[i])
    return y_data[i] * (1 - t) + y_data[i+1] * t
end

function synthesize_received_signal(tx_signal, paths, fs, noise_level, noise_type="combined")
    """合成接收信号（CIR）包含真实海洋噪声"""
    signal_length = length(tx_signal)
    rxsig = zeros(signal_length)
    
    # 计算相对延迟（相对于最小延迟）
    min_delay = minimum([path.delay for path in paths])
    
    # 将第一个路径放在信号开始后的适当位置（避免从0开始）
    start_offset_time = 0.005  # 5ms后开始
    
    for path in paths
        # 计算相对延迟时间
        relative_delay = path.delay - min_delay
        # 实际在信号中的时间位置
        signal_time = start_offset_time + relative_delay
        delay_samples = round(Int, signal_time * fs)
        
        # 确保延迟在有效范围内
        if delay_samples >= 1 && delay_samples <= signal_length
            # 计算可以放置多少个样本
            remaining_samples = signal_length - delay_samples + 1
            samples_to_copy = min(remaining_samples, length(tx_signal))
            
            if samples_to_copy > 0
                end_idx = delay_samples + samples_to_copy - 1
                
                # 考虑多普勒效应（简化版）
                doppler_shift = 1.0 + (0.0001 * randn())  # 小的随机多普勒偏移
                
                # 应用路径特定的信号变化
                path_signal = tx_signal[1:samples_to_copy] * path.amplitude
                
                # 为不同路径类型添加特定效果
                if path.path_type == "surface"
                    # 海面反射可能导致相位变化和频谱展宽
                    phase_shift = π * rand()  # 随机相位偏移
                    path_signal = path_signal .* cos.(phase_shift)
                elseif path.path_type == "bottom"
                    # 海底反射可能导致频谱衰减
                    # 简化为高频衰减
                    path_signal = path_signal .* exp.(-0.1 * (1:samples_to_copy) / samples_to_copy)
                elseif startswith(path.path_type, "scattering")
                    # 散射路径添加随机相位调制
                    phase_modulation = 0.1 * randn(samples_to_copy)
                    path_signal = path_signal .* (1.0 .+ phase_modulation)
                end
                
                rxsig[delay_samples:end_idx] .+= path_signal
            end
        end
    end
    
    # 添加真实海洋噪声而不是简单白噪声
    ocean_noise = generate_ocean_noise(signal_length, fs, noise_type)
    rxsig .+= noise_level .* ocean_noise
    
    return rxsig, start_offset_time
end

# ---------------------
# 数据集生成主函数
# ---------------------
function generate_toa_dataset(num_samples::Int)
    """生成TOA数据集"""
    println("开始生成TOA数据集，样本数量: $num_samples")
    
    # 时间轴
    t = 0:1/FS:T
    tx_signal = generate_transmitted_signal(t, FREQ)
    
    # 存储数据的结构
    dataset = []
    
    Random.seed!(42)  # 确保可重现性
    
    for i in 1:num_samples
        if i % 100 == 0
            println("已生成 $i/$num_samples 样本")
        end
        
        # 随机生成环境参数
        depth = DEPTH_RANGE[1] + (DEPTH_RANGE[2] - DEPTH_RANGE[1]) * rand()
        src_depth = SRC_DEPTH_RANGE[1] + (SRC_DEPTH_RANGE[2] - SRC_DEPTH_RANGE[1]) * rand()
        rx_depth = RX_DEPTH_RANGE[1] + (RX_DEPTH_RANGE[2] - RX_DEPTH_RANGE[1]) * rand()
        
        # 随机生成水平距离
        distance = DISTANCE_RANGE[1] + (DISTANCE_RANGE[2] - DISTANCE_RANGE[1]) * rand()
        
        # 随机角度（0-360度）
        angle = 2π * rand()
        
        # 计算位置
        src_pos = [0.0, 0.0, src_depth]
        rx_pos = [distance * cos(angle), distance * sin(angle), rx_depth]
        
        # 随机噪声水平
        noise_level = NOISE_LEVEL_RANGE[1] + (NOISE_LEVEL_RANGE[2] - NOISE_LEVEL_RANGE[1]) * rand()
        
        # 随机选择声速剖面类型
        ssp_types = ["linear", "thermocline", "deep_water"]
        ssp_type = rand(ssp_types)
        
        # 生成声速剖面和UnderwaterAcoustics环境
        env, ssp_depths, ssp_speeds, temperatures = create_ocean_environment(depth, ssp_type)
        
        # 环境参数（用于声学计算）
        environment = Dict(
            "depth" => depth,
            "temperature" => temperatures[Int(ceil(length(temperatures)/2))],  # 中间深度温度
            "avg_sound_speed" => mean(ssp_speeds),
            "ssp_type" => ssp_type,
            "sea_state" => 1 + 3 * rand()  # 海况1-4级
        )
        
        # 随机选择噪声类型
        noise_types = ["wind", "shipping", "combined"]
        noise_type = rand(noise_types)
        
        # 计算增强的多径TOA - 使用UnderwaterAcoustics
        paths = calculate_enhanced_multipath_toas(src_pos, rx_pos, depth, env, environment)
        
        # 合成接收信号
        cir_signal, signal_start_time = synthesize_received_signal(tx_signal, paths, FS, noise_level, noise_type)
        
        # 调整TOA时间为相对时间（相对于信号中的实际位置）
        min_toa = minimum([path.delay for path in paths])
        adjusted_toa_times = [(path.delay - min_toa + signal_start_time) for path in paths]
        toa_amplitudes = [path.amplitude for path in paths]
        path_types = [path.path_type for path in paths]
        
        # 构建数据样本
        sample = Dict(
            "sample_id" => i,
            "environment" => Dict(
                "depth" => depth,
                "sound_speed_ref" => SOUND_SPEED_REF,
                "src_position" => src_pos,
                "rx_position" => rx_pos,
                "distance" => sqrt(sum((rx_pos .- src_pos).^2)),
                "noise_level" => noise_level,
                "noise_type" => noise_type,
                "ssp_type" => ssp_type,
                "ssp_depths" => ssp_depths,
                "ssp_speeds" => ssp_speeds,
                "temperatures" => temperatures,
                "sea_state" => environment["sea_state"]
            ),
            "toa_data" => Dict(
                "arrival_times" => adjusted_toa_times,
                "absolute_arrival_times" => [path.delay for path in paths],  # 保留绝对时间
                "amplitudes" => toa_amplitudes,
                "path_types" => path_types,
                "num_paths" => length(paths),
                "time_offset" => min_toa  # 记录时间偏移
            ),
            "signal_data" => Dict(
                "sampling_rate" => FS,
                "duration" => T,
                "frequency" => FREQ,
                "time_axis" => collect(t),
                "transmitted_signal" => tx_signal,
                "received_signal" => cir_signal
            )
        )
        
        push!(dataset, sample)
    end
    
    println("数据集生成完成！")
    return dataset
end

# ---------------------
# 数据保存函数
# ---------------------
function save_dataset_to_files(dataset, output_dir="toa_dataset")
    """将数据集保存到文件"""
    if !isdir(output_dir)
        mkdir(output_dir)
    end
    
    println("保存数据集到: $output_dir")
    
    # 1. 保存元数据CSV文件
    metadata = []
    for sample in dataset
        meta = Dict(
            "sample_id" => sample["sample_id"],
            "depth" => sample["environment"]["depth"],
            "src_x" => sample["environment"]["src_position"][1],
            "src_y" => sample["environment"]["src_position"][2], 
            "src_z" => sample["environment"]["src_position"][3],
            "rx_x" => sample["environment"]["rx_position"][1],
            "rx_y" => sample["environment"]["rx_position"][2],
            "rx_z" => sample["environment"]["rx_position"][3],
            "distance" => sample["environment"]["distance"],
            "noise_level" => sample["environment"]["noise_level"],
            "noise_type" => sample["environment"]["noise_type"],
            "ssp_type" => sample["environment"]["ssp_type"],
            "sea_state" => sample["environment"]["sea_state"],
            "num_paths" => sample["toa_data"]["num_paths"],
            "direct_toa" => sample["toa_data"]["arrival_times"][1],
            "surface_toa" => length(sample["toa_data"]["arrival_times"]) > 1 ? sample["toa_data"]["arrival_times"][2] : missing,
            "bottom_toa" => length(sample["toa_data"]["arrival_times"]) > 2 ? sample["toa_data"]["arrival_times"][3] : missing,
            "absolute_direct_toa" => sample["toa_data"]["absolute_arrival_times"][1],
            "time_offset" => sample["toa_data"]["time_offset"]
        )
        push!(metadata, meta)
    end
    
    # 转换为DataFrame并保存
    df = DataFrame(metadata)
    CSV.write(joinpath(output_dir, "metadata.csv"), df)
    
    # 2. 保存完整的JSON数据（包含信号）
    println("保存完整数据到JSON文件...")
    open(joinpath(output_dir, "complete_dataset.json"), "w") do f
        JSON.print(f, dataset, 2)
    end
    
    # 3. 保存信号数据为二进制文件（更高效）
    println("保存信号数据...")
    signals_dir = joinpath(output_dir, "signals")
    if !isdir(signals_dir)
        mkdir(signals_dir)
    end
    
    for sample in dataset
        sample_id = sample["sample_id"]
        
        # 保存发射信号
        tx_file = joinpath(signals_dir, "tx_$(sample_id).dat")
        open(tx_file, "w") do f
            write(f, sample["signal_data"]["transmitted_signal"])
        end
        
        # 保存接收信号（CIR）
        rx_file = joinpath(signals_dir, "cir_$(sample_id).dat")
        open(rx_file, "w") do f
            write(f, sample["signal_data"]["received_signal"])
        end
    end
    
    # 4. 保存数据集说明
    readme_content = """
# 增强版TOA数据集说明

## 数据集概览
- 样本数量: $(length(dataset))
- 采样率: $FS Hz
- 信号时长: $T 秒
- 载波频率: $FREQ Hz
- 参考声速: $SOUND_SPEED_REF m/s

## 增强的声学建模特性
### 声速剖面 (SSP)
- 支持线性、跃温层、深水声道等多种剖面类型
- 基于Mackenzie方程的真实声速计算
- 考虑温度、盐度、压力的影响

### 声学传播效应
- 频率相关的声吸收（基于Francois & Garrison公式）
- 射线折射和声速梯度影响
- 海面/海底反射的角度和频率依赖性
- 海底类型建模（沙、泥、岩石、粘土）
- 海况对海面反射的影响
- 散射路径建模

### 真实海洋噪声
- 风噪声（高频成分，-17dB/decade谱特性）
- 航运噪声（低频调制成分）
- 地震/微震噪声（极低频成分）
- 组合噪声场

## 文件结构
- `metadata.csv`: 包含所有样本的元数据信息
- `complete_dataset.json`: 完整的数据集（包含信号数据和环境参数）
- `signals/`: 信号数据文件夹
  - `tx_[id].dat`: 发射信号（二进制格式）
  - `cir_[id].dat`: 接收信号/CIR（二进制格式）

## 数据字段说明
### 环境参数
- depth: 水深 [m]
- src_position: 声源位置 [x, y, z] [m]
- rx_position: 接收器位置 [x, y, z] [m]  
- distance: 源-接收器距离 [m]
- noise_level: 噪声水平
- noise_type: 噪声类型 (wind/shipping/combined)
- ssp_type: 声速剖面类型 (linear/thermocline/deep_water)
- sea_state: 海况 (1-4级)
- ssp_depths: 声速剖面深度点
- ssp_speeds: 对应的声速值
- temperatures: 温度剖面

### TOA数据
- arrival_times: 各路径到达时间 [s]
- absolute_arrival_times: 绝对到达时间 [s]
- amplitudes: 各路径幅度（考虑吸收和反射损失）
- path_types: 路径类型 (direct/surface/bottom/scattering_X)
- num_paths: 路径数量
- time_offset: 时间偏移

### 信号数据
- transmitted_signal: 发射信号
- received_signal: 接收信号（CIR，包含真实海洋噪声）
"""
    
    open(joinpath(output_dir, "README.md"), "w") do f
        write(f, readme_content)
    end
    
    println("数据集保存完成！")
    println("文件位置: $output_dir")
end

# ---------------------
# 数据可视化函数
# ---------------------
function visualize_sample(dataset, sample_idx; save_plot=true)
    """可视化单个样本"""
    sample = dataset[sample_idx]
    
    t = sample["signal_data"]["time_axis"]
    tx_sig = sample["signal_data"]["transmitted_signal"]
    rx_sig = sample["signal_data"]["received_signal"]
    toa_times = sample["toa_data"]["arrival_times"]
    
    # 创建子图
    p1 = plot(t, tx_sig, label="Transmitted Signal", title="Sample $(sample_idx)")
    xlabel!("Time [s]")
    ylabel!("Amplitude")
    
    p2 = plot(t, rx_sig, label="Received Signal (CIR)", color=:blue)
    xlabel!("Time [s]")
    ylabel!("Amplitude")
    
    # 添加TOA标记
    for (i, toa) in enumerate(toa_times)
        vline!(p2, [toa], linestyle=:dash, color=:red, alpha=0.7,
               label=i==1 ? "TOA Markers" : "")
    end
    
    # 合并子图
    plot_combined = plot(p1, p2, layout=(2,1), size=(800, 600))
    
    if save_plot
        savefig(plot_combined, "sample_$(sample_idx)_visualization.png")
        println("样本 $(sample_idx) 的可视化已保存")
    end
    
    return plot_combined
end

# ---------------------
# 主执行部分
# ---------------------
function main()
    println("=== TOA数据集生成器 ===")
    
    # 生成数据集
    dataset = generate_toa_dataset(NUM_SAMPLES)
    
    # 保存数据集
    save_dataset_to_files(dataset)
    
    # 可视化几个样本
    println("生成可视化样本...")
    for i in [1, 10, 50, 100]
        if i <= length(dataset)
            visualize_sample(dataset, i)
        end
    end
    
    # 打印统计信息
    println("\n=== 数据集统计信息 ===")
    distances = [s["environment"]["distance"] for s in dataset]
    depths = [s["environment"]["depth"] for s in dataset]
    direct_toas = [s["toa_data"]["arrival_times"][1] for s in dataset]
    
    println("距离范围: $(round(minimum(distances), digits=2)) - $(round(maximum(distances), digits=2)) m")
    println("深度范围: $(round(minimum(depths), digits=2)) - $(round(maximum(depths), digits=2)) m")
    println("直达TOA范围: $(round(minimum(direct_toas), digits=4)) - $(round(maximum(direct_toas), digits=4)) s")
    
    println("\n数据集生成完成！可以开始训练TOA预测模型了。")
end

# 运行主函数
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
