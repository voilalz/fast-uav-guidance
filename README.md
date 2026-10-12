# fast-uav-guidance

高速（约 600 km/h）小型固定翼无人机在 GNSS 拒止条件下的自主导航方案研究。

- [总体方案](docs/navigation-design.md)：架构、景象匹配/地形匹配、ESKF 融合、误差预算、仿真与验证计划
- [硬件方案](docs/hardware-design.md)：传感器、处理器、飞控、执行机构选型与电气连接
- [软件方案](docs/software-design.md)：MCU 固件与 Jetson 视觉服务架构、滤波器设计、协议、验证结果
- [导航代码](nav/)：嵌入式 C++17 / C99 导航核心库、闭环仿真与单元测试
- [误差预算脚本](tools/error_budget.py)：`python3 tools/error_budget.py`

## 快速开始

```bash
cd nav
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/nav_sim 1200 1
```
